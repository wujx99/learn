"""A compact PyTorch training program that runs with python or torchrun.

Examples:
    python train.py --device cpu
    torchrun --standalone --nproc-per-node=2 train.py --device cpu
    torchrun --standalone --nproc-per-node=gpu train.py --device cuda
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import random
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank micro-batch size.")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup(args: argparse.Namespace) -> tuple[torch.device, int, int, int, bool]:
    distributed = is_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda:
        if distributed and local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(local_rank if distributed else 0)
        device = torch.device("cuda", local_rank if distributed else 0)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return device, rank, local_rank, world_size, distributed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(seed: int) -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(-3, 3, 640).unsqueeze(1)
    noise = 0.1 * torch.randn(x.shape, generator=generator)
    y = 0.5 * x.square() - 2.0 * x + 1.0 + noise
    return TensorDataset(x[:512], y[:512]), TensorDataset(x[512:], y[512:])


def build_loaders(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    distributed: bool,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
    )

    # Rank-strided validation avoids DistributedSampler padding duplicate samples.
    val_indices = range(rank, len(val_dataset), world_size)
    val_subset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_sampler


def build_model() -> nn.Module:
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
    distributed: bool,
) -> float:
    model.train()
    loss_and_count = torch.zeros(2, device=device, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)

    for step, (features, targets) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)
        final_micro_batch = step + 1 == len(loader)
        should_step = (step + 1) % accumulation_steps == 0 or final_micro_batch

        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(loader) - group_start)
        sync_context = nullcontext()
        if isinstance(model, DDP) and not should_step:
            sync_context = model.no_sync()

        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler.is_enabled(),
            ):
                loss = loss_fn(model(features), targets)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()

        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_and_count[0] += loss.detach().double() * features.size(0)
        loss_and_count[1] += features.size(0)

    reduce_sum(loss_and_count, distributed)
    return (loss_and_count[0] / loss_and_count[1]).item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    distributed: bool,
) -> float:
    model.eval()
    loss_and_count = torch.zeros(2, device=device, dtype=torch.float64)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        loss = loss_fn(model(features), targets)
        loss_and_count[0] += loss.double() * features.size(0)
        loss_and_count[1] += features.size(0)

    reduce_sum(loss_and_count, distributed)
    return (loss_and_count[0] / loss_and_count[1]).item()


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    unwrap(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            # Keep metadata compatible with torch.load(weights_only=True).
            "args": serializable_args,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be at least 1")

    device, rank, local_rank, world_size, distributed = setup(args)
    try:
        seed_everything(args.seed)
        train_dataset, val_dataset = build_datasets(args.seed)
        train_loader, val_loader, train_sampler = build_loaders(
            train_dataset,
            val_dataset,
            args.batch_size,
            args.seed,
            rank,
            world_size,
            distributed,
        )

        model = build_model().to(device)
        if distributed:
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            model = DDP(model, **ddp_kwargs)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        amp_enabled = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        start_epoch = 0

        if args.resume is not None:
            start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, device)

        if rank == 0:
            global_batch = args.batch_size * args.accumulation_steps * world_size
            print(
                f"mode={'DDP' if distributed else 'single'} backend="
                f"{dist.get_backend() if distributed else 'none'} device={device.type} "
                f"world_size={world_size} global_batch={global_batch} amp={amp_enabled}"
            )

        checkpoint_path = args.output_dir / "checkpoint.pt"
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scaler,
                device,
                args.accumulation_steps,
                distributed,
            )
            val_loss = evaluate(model, val_loader, loss_fn, device, distributed)

            if rank == 0:
                print(f"epoch={epoch + 1:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
                save_checkpoint(checkpoint_path, epoch, model, optimizer, scaler, args)

            if distributed:
                dist.barrier()

        if rank == 0:
            print(f"checkpoint={checkpoint_path}")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

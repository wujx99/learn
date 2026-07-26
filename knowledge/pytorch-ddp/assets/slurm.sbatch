#!/usr/bin/env bash
# Generic template: adapt partition, account, modules, paths, and resources.
#SBATCH --job-name=pytorch-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
master_addr="${nodes[0]}"
master_port=29400
train_script="knowledge/pytorch-ddp/assets/train.py"

# srun starts one torchrun agent per node; each agent starts one worker per GPU.
srun torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc-per-node=4 \
  --rdzv-id="${SLURM_JOB_ID}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${master_addr}:${master_port}" \
  "${train_script}" \
  --device cuda \
  --output-dir "/shared/checkpoints/${SLURM_JOB_ID}"

#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${1:-artifacts}"
PRECISION="${PRECISION:-fp32}"
ONNX_PATH="${ARTIFACT_DIR}/demo_dynamic.onnx"
ENGINE_PATH="${ARTIFACT_DIR}/demo_${PRECISION}.plan"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec is not on PATH; install TensorRT or use an NVIDIA TensorRT container." >&2
  exit 1
fi
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Missing ${ONNX_PATH}; run export_onnx.py first." >&2
  exit 1
fi

precision_args=()
case "${PRECISION}" in
  fp32) ;;
  fp16)
    if trtexec --help 2>&1 | grep -q -- '--fp16'; then
      precision_args+=(--fp16)
    else
      echo "This trtexec has no --fp16 (TensorRT 11.x strong typing)." >&2
      echo "Cast/quantize the ONNX model offline first; that belongs to the quantization lesson." >&2
      exit 2
    fi
    ;;
  *) echo "PRECISION must be fp32 or fp16" >&2; exit 2 ;;
esac

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --minShapes=images:1x3x32x32 \
  --optShapes=images:8x3x32x32 \
  --maxShapes=images:16x3x32x32 \
  "${precision_args[@]}" \
  --profilingVerbosity=detailed \
  --skipInference

for batch_size in 1 8 16; do
  trtexec \
    --loadEngine="${ENGINE_PATH}" \
    --shapes="images:${batch_size}x3x32x32" \
    --warmUp=500 \
    --duration=3 \
    --useCudaGraph
done

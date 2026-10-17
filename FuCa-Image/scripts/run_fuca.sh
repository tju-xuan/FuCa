#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python || true)}"
CONDA_BIN="${CONDA_BIN:-conda}"
USE_CONDA="${USE_CONDA:-auto}"
FLUX_ENV="${FLUX_ENV:-fuca-flux}"
QWEN_ENV="${QWEN_ENV:-fuca-qwen}"
TARGET="${TARGET:-all}"  # all, flux, or qwen
LIMIT="${LIMIT:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/fuca_$STAMP}"

PROMPT_FILE="${PROMPT_FILE:-$ROOT/benchmarks/prompts/drawbench.txt}"
if [[ "$LIMIT" != "0" ]]; then
  mkdir -p "$RESULT_ROOT"
  PROMPT_LIMIT_FILE="$RESULT_ROOT/drawbench_first_${LIMIT}.txt"
  head -n "$LIMIT" "$PROMPT_FILE" > "$PROMPT_LIMIT_FILE"
  PROMPT_FILE="$PROMPT_LIMIT_FILE"
fi

FLUX_MODEL_PATH="${FLUX_MODEL_PATH:-$ROOT/models/FLUX.1-dev}"
QWEN_IMAGE_MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-$ROOT/models/Qwen-Image}"

FLUX_SLOW_ANCHORS="${FLUX_SLOW_ANCHORS:-0,1,2,3,4,6,8,11,15,20,25,31,38,46}"
FLUX_FAST_ANCHORS="${FLUX_FAST_ANCHORS:-0,1,2,3,4,6,11,19,30,44}"
QWEN_SLOW_ANCHORS="${QWEN_SLOW_ANCHORS:-0,1,2,3,4,6,8,11,15,20,25,31,38,46}"
QWEN_FAST_ANCHORS="${QWEN_FAST_ANCHORS:-0,1,2,3,4,6,11,19,30,44}"

mkdir -p "$RESULT_ROOT"
echo "[FuCa] root=$ROOT"
echo "[FuCa] result_root=$RESULT_ROOT"
echo "[FuCa] prompts=$PROMPT_FILE"
if [[ "$USE_CONDA" == "auto" ]]; then
  if [[ -x "$CONDA_BIN" ]]; then
    USE_CONDA=1
  else
    USE_CONDA=0
  fi
fi

run_python() {
  local env_name="$1"
  shift
  if [[ "$USE_CONDA" == "1" ]]; then
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$CONDA_BIN" run -n "$env_name" python "$@"
  else
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "No python interpreter found. Set PYTHON=python or enable CONDA_BIN." >&2
      exit 2
    fi
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
  fi
}

run_flux() {
  local flux_root="$RESULT_ROOT/flux"
  mkdir -p "$flux_root"

  run_python "$FLUX_ENV" "$ROOT/benchmarks/flux_drawbench/run_flux_dense.py" \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$FLUX_MODEL_PATH" \
    --output-dir "$flux_root/dense50" \
    --width 1024 \
    --height 1024 \
    --num-steps 50 \
    --seed 42 \
    --guidance-scale 3.5 \
    --method-name dense50

  run_python "$FLUX_ENV" "$ROOT/benchmarks/flux_drawbench/run_flux_fuca.py" \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$FLUX_MODEL_PATH" \
    --output-dir "$flux_root/fuca_slow" \
    --width 1024 \
    --height 1024 \
    --num-steps 50 \
    --seed 42 \
    --guidance-scale 3.5 \
    --anchor-indices "$FLUX_SLOW_ANCHORS" \
    --correction-start-anchor 4 \
    --alpha-clip 1.0 \
    --channel-groups 1

  run_python "$FLUX_ENV" "$ROOT/benchmarks/flux_drawbench/run_flux_fuca.py" \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$FLUX_MODEL_PATH" \
    --output-dir "$flux_root/fuca_fast" \
    --width 1024 \
    --height 1024 \
    --num-steps 50 \
    --seed 42 \
    --guidance-scale 3.5 \
    --anchor-indices "$FLUX_FAST_ANCHORS" \
    --correction-start-anchor 4 \
    --alpha-clip 1.0 \
    --channel-groups 1

}

run_qwen() {
  local qwen_root="$RESULT_ROOT/qwen"
  mkdir -p "$qwen_root"
  export QWEN_OFFLOAD_MODE="${QWEN_OFFLOAD_MODE:-none}"

  run_python "$QWEN_ENV" "$ROOT/benchmarks/qwen_drawbench/run_qwen_method.py" \
    --method dense50 \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$QWEN_IMAGE_MODEL_PATH" \
    --output-dir "$qwen_root/dense50" \
    --width 1664 \
    --height 928 \
    --num-steps 50 \
    --seed 0 \
    --true-cfg-scale 1.0 \
    --max-sequence-length 512

  run_python "$QWEN_ENV" "$ROOT/benchmarks/qwen_drawbench/run_qwen_method.py" \
    --method ours \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$QWEN_IMAGE_MODEL_PATH" \
    --output-dir "$qwen_root/fuca_slow" \
    --width 1664 \
    --height 928 \
    --num-steps 50 \
    --seed 0 \
    --true-cfg-scale 1.0 \
    --max-sequence-length 512 \
    --anchor-indices "$QWEN_SLOW_ANCHORS" \
    --correction-start-anchor 4 \
    --alpha-clip 1.0 \
    --channel-groups 1

  run_python "$QWEN_ENV" "$ROOT/benchmarks/qwen_drawbench/run_qwen_method.py" \
    --method ours \
    --prompt-file "$PROMPT_FILE" \
    --model-path "$QWEN_IMAGE_MODEL_PATH" \
    --output-dir "$qwen_root/fuca_fast" \
    --width 1664 \
    --height 928 \
    --num-steps 50 \
    --seed 0 \
    --true-cfg-scale 1.0 \
    --max-sequence-length 512 \
    --anchor-indices "$QWEN_FAST_ANCHORS" \
    --correction-start-anchor 4 \
    --alpha-clip 1.0 \
    --channel-groups 1

}

case "$TARGET" in
  all)
    run_flux
    run_qwen
    ;;
  flux)
    run_flux
    ;;
  qwen)
    run_qwen
    ;;
  *)
    echo "Unsupported TARGET=$TARGET. Use TARGET=all, TARGET=flux, or TARGET=qwen." >&2
    exit 2
    ;;
esac

echo "[FuCa] generated images=$RESULT_ROOT"

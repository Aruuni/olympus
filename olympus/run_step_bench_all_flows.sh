#!/usr/bin/env bash
# Run the multi-agent step-change benchmark for flows ∈ {1, 2, 3, 4}.
# Each pass writes its own output directory under olympus/data
# (`step_bench_n1`, `step_bench_n2`, ...). Extra args after the checkpoint
# are forwarded verbatim to step_change_benchmark.py — useful for things
# like `--repeats 3 --change-at 30 --base-bw 30 --base-delay 30`.
#
# Usage:
#   sudo -E env PATH="$PATH" HOME="$HOME" \
#     olympus/run_step_bench_all_flows.sh \
#     <path/to/mat_cwnd_model.pt> [extra_step_change_benchmark_flags...]

set -eu

CKPT="${1:?usage: $0 <checkpoint.pt> [extra_args...]}"
shift

CONFIG="olympus/config.yaml"
SCRIPT="olympus/step_change_benchmark.py"
PY="${PY:-./venv_training/bin/python}"

if [[ ! -f "$CKPT"      ]]; then echo "checkpoint not found: $CKPT" >&2; exit 1; fi
if [[ ! -f "$CONFIG"    ]]; then echo "config not found: $CONFIG"    >&2; exit 1; fi
if [[ ! -f "$SCRIPT"    ]]; then echo "benchmark not found: $SCRIPT" >&2; exit 1; fi

for N in 1 2 3 4; do
  RUN_NAME="step_bench_n${N}_$(date +%Y%m%d-%H%M%S)"
  echo
  echo "===== flows=${N}  run_name=${RUN_NAME} ====="
  mn -c >/dev/null 2>&1 || true
  "$PY" "$SCRIPT" \
      --config       "$CONFIG" \
      --checkpoint   "$CKPT" \
      --flows        "$N" \
      --run-name     "$RUN_NAME" \
      "$@"
done

echo
echo "All four flow-count benchmarks done. Output directories:"
ls -d "$(dirname "$CKPT")/../../"data/step_bench_n*_* 2>/dev/null \
  || ls -d olympus/data/step_bench_n*_* 2>/dev/null \
  || true

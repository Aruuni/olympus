#!/usr/bin/env bash
# Reset all training state — deletes checkpoint, logs, and episode plots.
# Run this before starting a fresh training run.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[reset] removing checkpoint..."
rm -f "$ROOT"/oc_model_test.pt

echo "[reset] removing training log..."
rm -f "$ROOT"/oc_model_test_log.csv

echo "[reset] removing state logs..."
rm -f /tmp/oc_state_ep*.csv

echo "[reset] removing episode plots..."
rm -f "$ROOT"/plots/ep*.pdf
rm -f "$ROOT"/plots/episode_returns.csv

echo "[reset] done — ready for fresh training run"

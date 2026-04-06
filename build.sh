#!/usr/bin/env bash
# Usage: sudo ./build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== astraea venv (python3.11 + TensorFlow) ==="
python3.11 -m venv "$ROOT/venv_astraea"
"$ROOT/venv_astraea/bin/pip" install --upgrade pip setuptools wheel
"$ROOT/venv_astraea/bin/pip" install -r "$ROOT/requirements.txt"
"$ROOT/venv_astraea/bin/python" setup.py build_ext --inplace

echo "=== training venv (python3.8 + PyTorch) ==="
python3.8 -m venv "$ROOT/venv_training"
"$ROOT/venv_training/bin/pip" install --upgrade pip setuptools wheel
"$ROOT/venv_training/bin/pip" install -r "$ROOT/training/requirements.txt"

echo "=== C binaries ==="
cc -O2 -Wall -Wextra -pthread -o "$ROOT/oc_listener"      "$ROOT/oc_listener.c"
cc -O2 -Wall -Wextra         -o "$ROOT/astraea_listener"  "$ROOT/astraea_listener.c"

echo "=== done ==="

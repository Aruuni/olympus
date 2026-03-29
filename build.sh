#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-$ROOT/venv_runtime}"
REQ_FILE="${2:-$ROOT/requirements.txt}"

echo "[1/5] create venv: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "[2/5] activate venv"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/5] upgrade packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "[4/5] install requirements from $REQ_FILE"
python -m pip install -r "$REQ_FILE"

echo "[5/5] build tcp_sockopt and listener"
python setup.py build_ext --inplace
cc -O2 -Wall -Wextra -o astraea_listener astraea_listener.c

echo
echo "done"
echo "activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "python: $(python -c 'import sys; print(sys.executable)')"
echo "listener: $ROOT/astraea_listener"
echo "tcp_sockopt:"
ls -1 "$ROOT"/tcp_sockopt*.so
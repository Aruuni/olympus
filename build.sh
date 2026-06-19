#!/usr/bin/env bash
# Usage: ./build.sh   (no sudo needed; this only builds, it does not load
#                      kernel modules — see setup.sh for that)
#
# Recreates both Python environments from scratch and builds the native bits:
#   astraea/venv_astraea  python3.11 + TensorFlow   (requirements.txt)
#   venv_training         python3.8  + PyTorch       (olympus/requirements.txt)
# The tcp_sockopt C extension is compiled and installed into BOTH venvs, since
# both the astraea and training stacks import it. Finally the C listeners are
# compiled. Existing venvs are removed first so the result is reproducible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ASTRAEA_VENV="$ROOT/astraea/venv_astraea"
TRAINING_VENV="$ROOT/venv_training"

echo "=== astraea venv (python3.11 + TensorFlow) ==="
rm -rf "$ASTRAEA_VENV"
python3.11 -m venv "$ASTRAEA_VENV"
"$ASTRAEA_VENV/bin/pip" install --upgrade pip setuptools wheel
"$ASTRAEA_VENV/bin/pip" install -r "$ROOT/requirements.txt"
echo "--- building tcp_sockopt into astraea venv ---"
"$ASTRAEA_VENV/bin/pip" install "$ROOT"

echo "=== training venv (python3.8 + PyTorch) ==="
rm -rf "$TRAINING_VENV"
python3.8 -m venv "$TRAINING_VENV"
"$TRAINING_VENV/bin/pip" install --upgrade pip setuptools wheel
"$TRAINING_VENV/bin/pip" install -r "$ROOT/olympus/requirements.txt"
echo "--- building tcp_sockopt into training venv ---"
"$TRAINING_VENV/bin/pip" install "$ROOT"

echo "=== C binaries ==="
cc -O2 -Wall -Wextra -pthread -o "$ROOT/oc_listener"      "$ROOT/oc_listener.c"
cc -O2 -Wall -Wextra         -o "$ROOT/astraea_listener"  "$ROOT/astraea_listener.c"

echo "=== done ==="
echo "astraea venv : $ASTRAEA_VENV"
echo "training venv: $TRAINING_VENV"

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cc -O2 -Wall -Wextra -pthread \
  -o "$HERE/oc_listener" \
  "$HERE/oc_listener.c"

echo "[build] wrote $HERE/oc_listener"

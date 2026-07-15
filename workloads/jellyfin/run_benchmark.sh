#!/usr/bin/env bash
set -euo pipefail

URL='https://jellyfin.phoenixremoteaccess.uk/web/#/details?id=762277cc91effea4c30f648c3c106797&serverId=676fc56ba97448b3a598c30f8be4a1ce'
LABEL="${1:-baseline-linux}"
DURATION="${DURATION:-180}"

exec .venv/bin/python jellyfin_client_benchmark.py \
  --url "$URL" \
  --auth auth.json \
  --duration "$DURATION" \
  --label "$LABEL" \
  --manual-play

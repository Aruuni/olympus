#!/usr/bin/env bash
set -u

URL='https://jellyfin.phoenixremoteaccess.uk/web/#/details?id=762277cc91effea4c30f648c3c106797&serverId=676fc56ba97448b3a598c30f8be4a1ce'

DURATION="${DURATION:-180}"
EXPERIMENT="${1:-two-stream}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p logs

echo "Starting two concurrent Jellyfin streams..."

python jellyfin_client_benchmark.py \
    --url "$URL" \
    --auth auth.json \
    --duration "$DURATION" \
    --label "${EXPERIMENT}-stream-1-${STAMP}" \
    --headless \
    > "logs/${EXPERIMENT}-stream-1-${STAMP}.log" 2>&1 &
PID1=$!

python jellyfin_client_benchmark.py \
    --url "$URL" \
    --auth auth.json \
    --duration "$DURATION" \
    --label "${EXPERIMENT}-stream-2-${STAMP}" \
    --headless \
    > "logs/${EXPERIMENT}-stream-2-${STAMP}.log" 2>&1 &
PID2=$!

cleanup() {
    kill "$PID1" "$PID2" 2>/dev/null || true
}

trap cleanup INT TERM

wait "$PID1"
STATUS1=$?

wait "$PID2"
STATUS2=$?

echo "Stream 1 exit status: $STATUS1"
echo "Stream 2 exit status: $STATUS2"
echo
echo "Logs:"
echo "  logs/${EXPERIMENT}-stream-1-${STAMP}.log"
echo "  logs/${EXPERIMENT}-stream-2-${STAMP}.log"

if [[ "$STATUS1" -ne 0 || "$STATUS2" -ne 0 ]]; then
    exit 1
fi
#!/usr/bin/env bash
set -Eeuo pipefail

TAILSCALE_IP="${TAILSCALE_IP:-100.90.202.72}"
FILE_PORT="${FILE_PORT:-8080}"
FILE_ROOT="${FILE_ROOT:-/srv/olympus-file-transfer}"
FILE_METRICS_LOG="${FILE_METRICS_LOG:-$FILE_ROOT/server_metrics.jsonl}"
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_ROOT/instrumented_file_server.py"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
    exit 1
fi

if [[ ! -d "$FILE_ROOT" ]]; then
    echo "File root does not exist: $FILE_ROOT" >&2
    echo "Run setup_file_server.sh first." >&2
    exit 1
fi

if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "Instrumented server script does not exist: $SERVER_SCRIPT" >&2
    exit 1
fi

if ! ip -4 addr show dev tailscale0 | grep -Fq "$TAILSCALE_IP/"; then
    echo "Expected Tailscale address $TAILSCALE_IP is not assigned to tailscale0." >&2
    exit 1
fi

if ss -ltn "sport = :$FILE_PORT" | grep -q LISTEN; then
    echo "TCP port $FILE_PORT is already in use." >&2
    exit 1
fi

current_cc="$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null || echo unknown)"
echo "Linux default TCP congestion control at server start: $current_cc"
echo "Serving $FILE_ROOT at http://$TAILSCALE_IP:$FILE_PORT/"
echo "Per-flow TCP_INFO metrics log: $FILE_METRICS_LOG"
exec python3 "$SERVER_SCRIPT" \
    --bind "$TAILSCALE_IP" \
    --port "$FILE_PORT" \
    --directory "$FILE_ROOT" \
    --metrics-log "$FILE_METRICS_LOG"

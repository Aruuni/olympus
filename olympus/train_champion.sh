#!/usr/bin/env bash
# Train the Proteus champion (seed) policy for the continual-learning
# exhibition.
#
# The champion is a recurrent SAC agent trained on olympus/proteus_champion.yaml,
# whose sweep deliberately excludes the exhibition's deployment regime
# (100 Mbps / 100 ms / 0.5x BDP). Its final checkpoint becomes both the CRL
# service's seed model and its frozen distillation teacher, and its saved
# replay buffer warm-starts the service.
#
# Usage:
#   sudo -E ./olympus/train_champion.sh            # 500 episodes (default)
#   sudo -E EPISODES=800 ./olympus/train_champion.sh
#   sudo -E SKIP_RESET=1 ./olympus/train_champion.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

CONFIG="$HERE/proteus_champion.yaml"
PY="$REPO_ROOT/venv_training/bin/python"
LISTENER="$HERE/oc_bridge"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "[champion] must run as root (mininet + raw socket control): sudo -E $0" >&2
  exit 1
fi

for path in "$CONFIG" "$PY"; do
  [[ -e "$path" ]] || { echo "[champion] missing: $path" >&2; exit 1; }
done

# The listener must be built locally; a foreign-glibc binary fails exec
# silently and episodes then run agentless against a frozen learner buffer.
if [[ ! -x "$LISTENER" ]]; then
  echo "[champion] building listener..."
  "$HERE/build_listener.sh"
fi

if [[ "${SKIP_RESET:-0}" != "1" ]]; then
  echo "[champion] resetting runtime state..."
  "$HERE/reset.sh"
fi

ARGS=(--config "$CONFIG")
if [[ -n "${EPISODES:-}" ]]; then
  ARGS+=(--episodes "$EPISODES")
fi

echo "[champion] config=$CONFIG"
echo "[champion] outputs=$HERE/models/proteus_champion"
exec "$PY" "$HERE/train.py" "${ARGS[@]}"

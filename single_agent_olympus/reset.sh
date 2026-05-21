#!/usr/bin/env bash
# All-encompassing runtime reset for single_agent_olympus.
#
# This script kills leftover training/runtime processes and cleans Mininet
# residue. It intentionally does NOT delete checkpoints, run folders, episode
# traces, plots, telemetry, or learner logs.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

# Known learner ports in this repo:
#   6001 legacy Option-Critic
#   6002 td3_clean_slate legacy
#   6101 recurrent_ppo_clean_slate
#   6201 orca
#   6301 single_agent_olympus default
LEARNER_PORTS=(6001 6002 6101 6201 6301)

run_sudo() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$@" 2>/dev/null || true
  else
    sudo "$@" 2>/dev/null || true
  fi
}

kill_pattern() {
  local pattern="$1"
  run_sudo pkill -KILL -f "$pattern"
}

kill_exact() {
  local name="$1"
  run_sudo pkill -KILL -x "$name"
}

echo "[reset] killing single_agent_olympus processes..."
if [[ "${SAO_RESET_SKIP_ORCHESTRATOR:-0}" != "1" ]]; then
  kill_pattern "$ROOT/orchestrator.py"
fi
kill_pattern "$ROOT/algorithms/.*/worker.py"
kill_pattern "$ROOT/algorithms/.*/learner.py"
if [[ "${SAO_RESET_SKIP_ORCHESTRATOR:-0}" != "1" ]]; then
  kill_pattern "single_agent_olympus/orchestrator.py"
fi
kill_pattern "single_agent_olympus/algorithms/.*/worker.py"
kill_pattern "single_agent_olympus/algorithms/.*/learner.py"

echo "[reset] killing listener processes..."
kill_pattern "$REPO_ROOT/orca/astraea_listener"
kill_pattern "$REPO_ROOT/recurrent_ppo_clean_slate/astraea_listener"
kill_pattern "$REPO_ROOT/td3_clean_slate/astraea_listener"
kill_pattern "astraea_listener"
kill_pattern "oc_listener"

echo "[reset] killing legacy learner/worker/orchestrator processes..."
kill_pattern "orca/worker.py"
kill_pattern "orca/learner.py"
kill_pattern "orca/orchestrator.py"
kill_pattern "recurrent_ppo_clean_slate/worker.py"
kill_pattern "recurrent_ppo_clean_slate/learner.py"
kill_pattern "recurrent_ppo_clean_slate/orchestrator.py"
kill_pattern "td3_clean_slate/worker.py"
kill_pattern "td3_clean_slate/learner.py"
kill_pattern "td3_clean_slate/orchestrator.py"

echo "[reset] killing traffic/test leftovers..."
kill_exact iperf3
kill_exact iperf
kill_exact ss
kill_exact mnexec

echo "[reset] freeing learner ports..."
for port in "${LEARNER_PORTS[@]}"; do
  run_sudo fuser -k "${port}/tcp"
done

sleep 1

echo "[reset] mininet cleanup..."
run_sudo mn -c

echo "[reset] preserving checkpoints and run data under:"
echo "        $ROOT/data"
echo "[reset] done"

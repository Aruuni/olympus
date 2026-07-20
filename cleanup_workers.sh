#!/usr/bin/env bash
# Kill orphaned Olympus worker/listener processes left behind by dead runs.
#
# A process is considered a dead worker only if BOTH hold:
#   1. its command line references this repo (worker.py, deployment.server,
#      train.py, astraea_listener, oc_listener, ...)
#   2. its parent is init (PPID 1) — the orchestrator/benchmark that spawned
#      it is gone. Workers of a live run have a live parent and are skipped.
#
# Usage:
#   ./cleanup_workers.sh            # kill orphans (TERM, then KILL stragglers)
#   ./cleanup_workers.sh --dry-run  # only list what would be killed
#   ./cleanup_workers.sh --mn       # additionally run 'mn -c' afterwards
#
# Root-owned orphans (runs launched under sudo) need sudo to kill; the
# script uses it automatically when required.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
CLEAN_MN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --mn)      CLEAN_MN=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# pid ppid user cmd for orphaned processes whose cmdline points into this repo
find_orphans() {
    ps -eo pid=,ppid=,user:32=,args= | awk -v repo="$REPO_DIR" '
        $2 == 1 && index($0, repo) && !/awk|cleanup_workers/ {
            cmd = ""
            for (i = 4; i <= NF; i++) cmd = cmd (i > 4 ? " " : "") $i
            print $1 "\t" $3 "\t" cmd
        }'
}

ORPHANS="$(find_orphans)"
if [ -z "$ORPHANS" ]; then
    echo "No orphaned Olympus processes found."
else
    COUNT=$(printf '%s\n' "$ORPHANS" | wc -l)
    echo "Found $COUNT orphaned Olympus process(es):"
    printf '%s\n' "$ORPHANS" | while IFS=$'\t' read -r pid user cmd; do
        printf '  pid %-8s %-8s %.120s\n' "$pid" "$user" "$cmd"
    done

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Dry run: nothing killed."
    else
        SUDO=""
        if [ "$(id -u)" -ne 0 ] && printf '%s\n' "$ORPHANS" | awk -F'\t' -v me="$(id -un)" '$2 != me {found=1} END {exit !found}'; then
            SUDO="sudo"
            echo "Some orphans belong to other users; using sudo."
        fi

        PIDS=$(printf '%s\n' "$ORPHANS" | cut -f1)
        # shellcheck disable=SC2086
        $SUDO kill $PIDS 2>/dev/null
        sleep 3

        LEFT=""
        for pid in $PIDS; do
            kill -0 "$pid" 2>/dev/null && LEFT="$LEFT $pid"
        done
        if [ -n "$LEFT" ]; then
            echo "Escalating to SIGKILL for:$LEFT"
            # shellcheck disable=SC2086
            $SUDO kill -9 $LEFT 2>/dev/null
        fi

        REMAINING="$(find_orphans)"
        if [ -z "$REMAINING" ]; then
            echo "Done: all orphaned Olympus processes are gone."
        else
            echo "WARNING: some processes survived:" >&2
            printf '%s\n' "$REMAINING" >&2
            exit 1
        fi
    fi
fi

if [ "$CLEAN_MN" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "Dry run: would run 'mn -c' to clear mininet state."
    else
        echo "Running 'mn -c' to clear mininet state..."
        sudo mn -c >/dev/null 2>&1 && echo "mininet state cleared." || echo "WARNING: 'mn -c' failed." >&2
    fi
fi

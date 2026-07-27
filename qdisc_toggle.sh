#!/usr/bin/env bash
# Swap an interface's fq_codel for a plain FIFO and put it back.
#
#   sudo ./qdisc_toggle.sh off eno1     # fq_codel -> pfifo (no AQM, no DRR)
#   sudo ./qdisc_toggle.sh on  eno1     # restore the saved qdisc setup
#   ./qdisc_toggle.sh status eno1       # show the qdisc tree (no sudo needed)
#
# Multiqueue NICs come up as `mq` at handle 0: with one fq_codel per hardware
# TX queue. Children under handle 0: are unaddressable ("Failed to find
# specified qdisc"), so `off` re-roots mq at handle 1: and then replaces each
# queue. `off` snapshots `tc qdisc show` first; `on` deletes the root so the
# kernel rebuilds the default tree, and only if that does not match the
# snapshot does it replay the recorded per-queue parameters.
set -euo pipefail

STATE_DIR="${QDISC_STATE_DIR:-/var/lib/olympus-qdisc}"
FIFO_QDISC="${FIFO_QDISC:-pfifo}"
FIFO_LIMIT="${FIFO_LIMIT:-1000}"
ROOT_HANDLE="${ROOT_HANDLE:-1:}"

usage() {
    echo "usage: $0 {off|on|status} <interface>" >&2
    exit 2
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "error: '$1' changes qdiscs and must run as root (use sudo)" >&2
        exit 1
    fi
}

state_file() { echo "$STATE_DIR/$1.qdisc"; }

root_kind() {
    tc qdisc show dev "$1" | awk '$0 ~ / root/ { print $2; exit }'
}

tx_queue_count() {
    local count
    count=$(find "/sys/class/net/$1/queues" -maxdepth 1 -name 'tx-*' | wc -l)
    echo "$((count > 0 ? count : 1))"
}

# Queue minors in the saved snapshot, e.g. "4 3 2 1".
saved_minors() {
    awk '{
        for (i = 1; i < NF; i++)
            if ($i == "parent") { sub(/^.*:/, "", $(i + 1)); print $(i + 1); break }
    }' "$1"
}

# "<kind> <options>" per saved child, with tc-printed packet suffixes stripped
# so the values can be fed straight back to tc.
saved_child_args() {
    awk -v want="$2" '{
        for (i = 1; i < NF; i++) {
            if ($i == "parent") {
                minor = $(i + 1); sub(/^.*:/, "", minor)
                if (minor != want) next
                line = $2
                for (j = i + 2; j <= NF; j++) {
                    field = $j
                    if (field ~ /^[0-9]+p$/) sub(/p$/, "", field)
                    line = line " " field
                }
                print line
                exit
            }
        }
    }' "$1"
}

# Handle-independent view of a qdisc tree, for comparing before and after.
canonical() {
    sed -e 's/^qdisc //' -e 's/[0-9a-f]*: //g' -e 's/ refcnt [0-9]*//' \
        -e 's/parent [0-9a-f]*:/parent :/' -e 's/[[:space:]]*$//' \
        | sort
}

cmd_status() {
    tc qdisc show dev "$1"
}

cmd_off() {
    local iface=$1 state kind queues i
    state=$(state_file "$iface")
    kind=$(root_kind "$iface")

    if [[ -e $state ]]; then
        echo "note: $state already exists; keeping the original snapshot"
    else
        mkdir -p "$STATE_DIR"
        tc qdisc show dev "$iface" > "$state"
        echo "saved current qdiscs to $state"
    fi

    if [[ $kind == "mq" || $kind == "mqprio" ]]; then
        queues=$(tx_queue_count "$iface")
        # Children of a 0:-handled root cannot be addressed; re-root mq at a
        # real handle. This briefly recreates the children as the system
        # default qdisc before they are replaced below.
        tc qdisc replace dev "$iface" root handle "$ROOT_HANDLE" mq
        for ((i = 1; i <= queues; i++)); do
            tc qdisc replace dev "$iface" parent "${ROOT_HANDLE}${i}" \
                "$FIFO_QDISC" limit "$FIFO_LIMIT"
        done
        echo "dev $iface: $FIFO_QDISC limit $FIFO_LIMIT on $queues queues (under mq $ROOT_HANDLE)"
    else
        tc qdisc replace dev "$iface" root "$FIFO_QDISC" limit "$FIFO_LIMIT"
        echo "dev $iface: $FIFO_QDISC limit $FIFO_LIMIT at root"
    fi
    tc qdisc show dev "$iface"
}

cmd_on() {
    local iface=$1 state kind minor args
    state=$(state_file "$iface")
    if [[ ! -f $state ]]; then
        echo "error: no saved state at $state; run '$0 off $iface' first" >&2
        exit 1
    fi

    # Dropping the root makes the kernel rebuild the boot-time default tree
    # (mq + net.core.default_qdisc), which is usually exactly the snapshot.
    tc qdisc del dev "$iface" root 2>/dev/null || true

    if diff -q <(canonical < "$state") \
               <(tc qdisc show dev "$iface" | canonical) >/dev/null; then
        echo "restored dev $iface to the saved default tree"
    else
        echo "default tree differs from the snapshot; replaying saved parameters"
        kind=$(awk '$0 ~ / root/ { print $2; exit }' "$state")
        if [[ $kind == "mq" || $kind == "mqprio" ]]; then
            tc qdisc replace dev "$iface" root handle "$ROOT_HANDLE" "$kind"
            for minor in $(saved_minors "$state"); do
                args=$(saved_child_args "$state" "$minor")
                [[ -n $args ]] || continue
                # shellcheck disable=SC2086 # args is a deliberate list
                tc qdisc replace dev "$iface" \
                    parent "${ROOT_HANDLE}${minor}" $args
            done
            echo "note: mq root is now handle $ROOT_HANDLE, not 0:;" \
                 "reboot or 'tc qdisc del dev $iface root' to get 0: back"
        else
            args=$(awk '$0 ~ / root/ {
                line = $2
                for (j = 3; j <= NF; j++) {
                    if ($j == "root" || $j == "dev" || $j ~ /:$/) continue
                    if ($j == $5) continue
                    field = $j
                    if (field ~ /^[0-9]+p$/) sub(/p$/, "", field)
                    line = line " " field
                }
                print line; exit
            }' "$state")
            # shellcheck disable=SC2086 # args is a deliberate list
            tc qdisc replace dev "$iface" root $args
        fi
    fi

    rm -f "$state"
    tc qdisc show dev "$iface"
}

[[ $# -eq 2 ]] || usage
action=$1
iface=$2

if [[ ! -d /sys/class/net/$iface ]]; then
    echo "error: no such interface: $iface" >&2
    exit 1
fi

case $action in
    off) require_root off; cmd_off "$iface" ;;
    on)  require_root on;  cmd_on  "$iface" ;;
    status) cmd_status "$iface" ;;
    *) usage ;;
esac

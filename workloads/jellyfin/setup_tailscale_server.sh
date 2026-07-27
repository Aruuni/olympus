#!/usr/bin/env bash
set -Eeuo pipefail

TAILSCALE_IP="${TAILSCALE_IP:-100.90.202.72}"
JELLYFIN_PORT="${JELLYFIN_PORT:-8096}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script with sudo." >&2
    exit 1
fi

if ! ip link show tailscale0 >/dev/null 2>&1; then
    echo "tailscale0 is not available." >&2
    exit 1
fi

if ! ip -4 addr show dev tailscale0 | grep -Fq "$TAILSCALE_IP/"; then
    echo "Expected Tailscale address $TAILSCALE_IP is not assigned to tailscale0." >&2
    exit 1
fi

if ! ss -ltn "sport = :$JELLYFIN_PORT" | grep -q LISTEN; then
    echo "Jellyfin is not listening on TCP port $JELLYFIN_PORT." >&2
    exit 1
fi

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
    # A broad deny for 8096 may already exist. Put the interface-specific
    # exception first so UFW evaluates it before any later deny rule.
    ufw --force delete allow in on tailscale0 to any port "$JELLYFIN_PORT" proto tcp \
        >/dev/null 2>&1 || true
    ufw insert 1 allow in on tailscale0 to any port "$JELLYFIN_PORT" proto tcp
    echo "Inserted the persistent Tailscale exception as UFW rule 1."
else
    if ! iptables -C INPUT -i tailscale0 -p tcp --dport "$JELLYFIN_PORT" -j ACCEPT 2>/dev/null; then
        iptables -I INPUT 1 -i tailscale0 -p tcp --dport "$JELLYFIN_PORT" -j ACCEPT
    fi
    echo "Added an iptables rule for Jellyfin on tailscale0."
    if command -v netfilter-persistent >/dev/null 2>&1; then
        netfilter-persistent save
        echo "Saved the firewall rule with netfilter-persistent."
    else
        echo "Warning: netfilter-persistent is unavailable; this firewall rule may need to be restored after reboot." >&2
    fi
fi

echo "Jellyfin is available inside the tailnet at http://$TAILSCALE_IP:$JELLYFIN_PORT/"

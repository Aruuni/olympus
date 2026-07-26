#!/usr/bin/env bash
set -Eeuo pipefail

TAILSCALE_IP="${TAILSCALE_IP:-100.90.202.72}"
FILE_PORT="${FILE_PORT:-8080}"
FILE_ROOT="${FILE_ROOT:-/srv/olympus-file-transfer}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this setup script with sudo." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required." >&2
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

install -d -m 0755 "$FILE_ROOT"

create_file() {
    local name="$1"
    local mib="$2"
    local target="$FILE_ROOT/$name"
    local partial="$target.partial"
    local expected_bytes=$((mib * 1024 * 1024))

    if [[ -e "$target" ]]; then
        local actual_bytes
        actual_bytes="$(stat -c '%s' "$target")"
        if [[ "$actual_bytes" -ne "$expected_bytes" ]]; then
            echo "$target exists but is $actual_bytes bytes; expected $expected_bytes." >&2
            exit 1
        fi
        echo "Keeping existing $target ($mib MiB)."
        return
    fi

    if [[ -e "$partial" ]]; then
        echo "Refusing to overwrite leftover partial file $partial." >&2
        exit 1
    fi

    echo "Creating $target ($mib MiB)..."
    dd if=/dev/zero of="$partial" bs=1M count="$mib" conv=fsync status=progress
    chmod 0644 "$partial"
    mv "$partial" "$target"
}

create_file "file-100MiB.bin" 100
create_file "file-500MiB.bin" 500
create_file "file-1GiB.bin" 1024

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
    ufw --force delete allow in on tailscale0 to any port "$FILE_PORT" proto tcp \
        >/dev/null 2>&1 || true
    ufw insert 1 allow in on tailscale0 to any port "$FILE_PORT" proto tcp
    echo "Inserted the persistent Tailscale exception as UFW rule 1."
else
    echo "UFW is not active; no firewall rule was changed."
fi

echo
echo "Files are ready under $FILE_ROOT:"
find "$FILE_ROOT" -maxdepth 1 -type f -printf '  %f  %s bytes\n' | sort
echo
echo "Select the congestion-control approach, then start the server with:"
echo "  sudo ./serve_file_server.sh"

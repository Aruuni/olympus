#!/usr/bin/env bash
# Usage: sudo ./build.sh
#
# Recreates both Python environments from scratch and builds the native bits:
#   astraea/venv_astraea  python3.11 + TensorFlow   (requirements.txt)
#   venv_training         python3.8  + PyTorch       (olympus/requirements.txt)
# The tcp_sockopt C extension is compiled and installed into BOTH venvs, since
# both the astraea and training stacks import it. Finally the C listeners are
# compiled. Existing venvs are removed first so the result is reproducible.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

INSTALL_USER="${SUDO_USER:-$(id -un)}"
INSTALL_HOME="$(getent passwd "$INSTALL_USER" | cut -d: -f6)"
[[ -n "$INSTALL_HOME" ]] || INSTALL_HOME="$HOME"

APT_PACKAGES=(
    mininet
)

ASTRAEA_VENV="$ROOT/astraea/venv_astraea"
TRAINING_VENV="$ROOT/venv_training"
SIM_BUILD_ALL="$ROOT/olympus/environments/raynet/sim/build_all.sh"
SIM_SUBMODULES=(
    "$ROOT/olympus/environments/raynet/sim/omnetpp"
    "$ROOT/olympus/environments/raynet/sim/inet4.5"
    "$ROOT/olympus/environments/raynet/sim/tcpPaced"
    "$ROOT/olympus/environments/raynet/sim/cubic"
    "$ROOT/olympus/environments/raynet/sim/raynet"
)

run_as_install_user() {
    if [[ "$(id -un)" == "$INSTALL_USER" ]]; then
        "$@"
    else
        sudo -H -u "$INSTALL_USER" env HOME="$INSTALL_HOME" "$@"
    fi
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local suffix
    local reply

    if [[ "$default" == "y" ]]; then
        suffix="[Y/n]"
    else
        suffix="[y/N]"
    fi

    if [[ ! -t 0 ]]; then
        [[ "$default" == "y" ]]
        return
    fi

    while true; do
        read -r -p "$prompt $suffix " reply
        reply="${reply:-$default}"
        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done
}

install_system_packages_if_needed() {
    local missing=()
    local package
    local sudo_cmd=()

    command -v apt-get >/dev/null 2>&1 || return 0
    command -v dpkg-query >/dev/null 2>&1 || return 0

    for package in "${APT_PACKAGES[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
            missing+=("$package")
        fi
    done

    ((${#missing[@]} == 0)) && return 0

    if ((EUID != 0)); then
        command -v sudo >/dev/null 2>&1 || {
            echo "ERROR: Missing packages: ${missing[*]}; sudo is required to install them" >&2
            exit 1
        }
        sudo_cmd=(sudo)
    fi

    echo "=== system packages: ${missing[*]} ==="
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y "${missing[@]}"
}

clean_python_build_artifacts() {
    rm -rf "$ROOT/build" "$ROOT/tcp_sockopt.egg-info"
}

sim_submodules_missing() {
    local dir
    for dir in "${SIM_SUBMODULES[@]}"; do
        [[ -e "$dir/.git" ]] || return 0
    done
    return 1
}

ensure_sim_submodules() {
    if ! sim_submodules_missing; then
        return 0
    fi

    echo "=== simulation submodules ==="
    if ask_yes_no "Some RayNet simulation submodules are missing. Initialize them now?" "y"; then
        run_as_install_user git submodule update --init --recursive -- "${SIM_SUBMODULES[@]#$ROOT/}"
    else
        echo "Skipping RayNet simulation stack because required submodules are missing."
        return 1
    fi
}

install_system_packages_if_needed

echo "=== astraea venv (python3.11 + TensorFlow) ==="
rm -rf "$ASTRAEA_VENV"
run_as_install_user python3.11 -m venv "$ASTRAEA_VENV"
run_as_install_user "$ASTRAEA_VENV/bin/pip" install --upgrade pip setuptools wheel
run_as_install_user "$ASTRAEA_VENV/bin/pip" install -r "$ROOT/requirements.txt"
echo "--- building tcp_sockopt into astraea venv ---"
clean_python_build_artifacts
run_as_install_user "$ASTRAEA_VENV/bin/pip" install "$ROOT"

echo "=== training venv (python3.8 + PyTorch) ==="
rm -rf "$TRAINING_VENV"
run_as_install_user python3.8 -m venv "$TRAINING_VENV"
run_as_install_user "$TRAINING_VENV/bin/pip" install --upgrade pip setuptools wheel
run_as_install_user "$TRAINING_VENV/bin/pip" install -r "$ROOT/olympus/requirements.txt"
run_as_install_user "$TRAINING_VENV/bin/pip" install mininet
echo "--- building tcp_sockopt into training venv ---"
clean_python_build_artifacts
run_as_install_user "$TRAINING_VENV/bin/pip" install "$ROOT"

echo "=== C binaries ==="
run_as_install_user cc -O2 -Wall -Wextra -pthread -o "$ROOT/oc_listener"      "$ROOT/oc_listener.c"
run_as_install_user cc -O2 -Wall -Wextra         -o "$ROOT/astraea_listener"  "$ROOT/astraea_listener.c"

echo "=== done ==="
echo "astraea venv : $ASTRAEA_VENV"
echo "training venv: $TRAINING_VENV"

echo "=== RayNet simulation stack ==="
if ensure_sim_submodules; then
    run_as_install_user "$SIM_BUILD_ALL"
fi

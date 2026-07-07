#!/usr/bin/env bash
#
# Build / rebuild the full RayNet + OMNeT++ simulation stack that is vendored
# under this directory as git submodules:
#
#   sim/omnetpp    OMNeT++ 6.x        (headless: Cmdenv only)
#   sim/inet4.5    INET 4.5 fork      -> symlinked into omnetpp/samples/inet4.5
#   sim/tcpPaced   pacing TCP ext     -> symlinked into omnetpp/samples/tcpPaced
#   sim/cubic      CUBIC ext          -> symlinked into omnetpp/samples/cubic
#   sim/raynet     RayNet simlibs + omnetbind pybind module + Python venv
#
# The script is self-locating: it derives all paths from its own location, so it
# works no matter where the Olympus repo is checked out. Build order encodes the
# dependency chain: omnetpp -> inet -> tcpPaced -> cubic -> raynet.
#
# Usage:
#   ./build_all.sh                 full incremental build of the whole stack
#   ./build_all.sh -r|--rebuild    clean-rebuild RayNet only (fastest after
#                                  editing simlibs / omnetbind / CleanSlate.cc)
#   ./build_all.sh -c|--clean-all  clean and rebuild EVERYTHING from scratch
#   ./build_all.sh -s|--stage S    build a single stage:
#                                  omnet | inet | ext | venv | raynet
#   ./build_all.sh -j N            parallel jobs (default: nproc)
#   ./build_all.sh -h|--help
#
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# Self-locating paths
# --------------------------------------------------------------------------- #
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export OMNET_PATH="$SIM_DIR/omnetpp"
export RAYNET_PATH="$SIM_DIR/raynet"
export INET_PATH="$OMNET_PATH/samples/inet4.5"
export TCPPACED_PATH="$OMNET_PATH/samples/tcpPaced"
export CUBIC_PATH="$OMNET_PATH/samples/cubic"
export RAYNET_VENV_PATH="$RAYNET_PATH/.venv"

JOBS="$(nproc)"
STAGE=""
REBUILD="false"
CLEAN_ALL="false"

log() { echo -e "\n=== [build_all] $* ($(date '+%H:%M:%S')) ==="; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--rebuild)   REBUILD="true"; shift ;;
        -c|--clean-all) CLEAN_ALL="true"; shift ;;
        -s|--stage)     STAGE="${2:-}"; shift 2 ;;
        -j)             JOBS="${2:-}"; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,40p'; exit 0 ;;
        *)              die "unknown option: $1 (try --help)" ;;
    esac
done

# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
[[ -d "$OMNET_PATH"  ]] || die "OMNeT++ submodule missing: $OMNET_PATH (run: git submodule update --init)"
[[ -d "$RAYNET_PATH" ]] || die "RayNet submodule missing: $RAYNET_PATH  (run: git submodule update --init)"
command -v python3.12 >/dev/null 2>&1 || die "python3.12 not found (RayNet requires it)"

# Ensure the sample symlinks exist (inet/tcpPaced/cubic live as flat submodules
# and are symlinked into omnetpp/samples/ so the build finds them there).
ensure_symlinks() {
    for name in inet4.5 tcpPaced cubic; do
        local link="$OMNET_PATH/samples/$name"
        if [[ ! -e "$link" ]]; then
            log "restoring missing symlink samples/$name -> ../../$name"
            ln -s "../../$name" "$link"
        fi
    done
}

# --------------------------------------------------------------------------- #
# Stage: OMNeT++ (headless configure + make)
# --------------------------------------------------------------------------- #
build_omnet() {
    log "OMNeT++: configure + make -j$JOBS"
    cd "$OMNET_PATH"
    [[ -f configure.user ]] || cp configure.user.dist configure.user
    # Headless: no Qt/OSG GUI, no OMNeT++ Python bindings (RayNet embeds it via pybind).
    sed -i 's/^WITH_QTENV=.*/WITH_QTENV=no/'                         configure.user || true
    sed -i 's/^WITH_OSG=.*/WITH_OSG=no/'                             configure.user || true
    sed -i 's/^WITH_OSGEARTH=.*/WITH_OSGEARTH=no/'                   configure.user || true
    sed -i 's/^WITH_SCAVE_PYTHON_BINDINGS=.*/WITH_SCAVE_PYTHON_BINDINGS=no/' configure.user || true
    sed -i 's/^WITH_PYTHON=.*/WITH_PYTHON=no/'                       configure.user || true
    set +u; source ./setenv; set -u
    [[ "$CLEAN_ALL" == "true" ]] && make cleanall || true
    ./configure
    make -j"$JOBS"
    which opp_run opp_makemake >/dev/null || die "OMNeT++ tools not on PATH after build"
}

# --------------------------------------------------------------------------- #
# Stage: INET
# --------------------------------------------------------------------------- #
build_inet() {
    ensure_symlinks
    log "INET: make makefiles + make -j$JOBS MODE=release"
    set +u; source "$OMNET_PATH/setenv"; set -u
    cd "$INET_PATH"
    set +u; source ./setenv; set -u
    [[ "$CLEAN_ALL" == "true" ]] && make clean MODE=release || true
    make makefiles
    make -j"$JOBS" MODE=release
    find . -name 'libINET*.so' -print -quit | grep -q . || die "libINET not produced"
}

# --------------------------------------------------------------------------- #
# Stage: external extensions (tcpPaced then cubic; regenerates makefiles so the
# INET/tcpPaced include+link paths are correct for this checkout)
# --------------------------------------------------------------------------- #
build_ext() {
    ensure_symlinks
    set +u; source "$OMNET_PATH/setenv"; set -u

    log "extension: tcpPaced (links INET)"
    cd "$TCPPACED_PATH"
    find . \( -name '*_m.cc' -o -name '*_m.h' \) -delete
    rm -rf out
    ( cd src && opp_makemake -f --deep --make-so -o tcpPaced -O out \
        -KINET_PROJ="$INET_PATH" -DINET_IMPORT \
        -I"$INET_PATH/src" -L"$INET_PATH/src" -lINET )
    make -j"$JOBS" MODE=release
    [[ -f src/libtcpPaced.so ]] || die "libtcpPaced.so not produced"

    log "extension: cubic (links INET + tcpPaced)"
    cd "$CUBIC_PATH"
    find . \( -name '*_m.cc' -o -name '*_m.h' \) -delete
    rm -rf out
    ( cd src && opp_makemake -f --deep --make-so -o cubic -O out \
        -KINET_PROJ="$INET_PATH" -DINET_IMPORT \
        -I"$INET_PATH/src" -L"$INET_PATH/src" -lINET \
        -I"$TCPPACED_PATH/src" -L"$TCPPACED_PATH/src" -ltcpPaced )
    make -j"$JOBS" MODE=release
    [[ -f src/libcubic.so ]] || die "libcubic.so not produced"
}

# --------------------------------------------------------------------------- #
# Stage: Python venv (recreate cleanly; the moved venv had stale absolute paths)
# --------------------------------------------------------------------------- #
build_venv() {
    log "RayNet venv: (re)creating $RAYNET_VENV_PATH with python3.12"
    rm -rf "$RAYNET_VENV_PATH"
    python3.12 -m venv "$RAYNET_VENV_PATH"
    # shellcheck disable=SC1091
    source "$RAYNET_VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -r "$RAYNET_PATH/requirements-extra.txt"
    deactivate
}

# --------------------------------------------------------------------------- #
# Stage: RayNet simlibs + omnetbind (delegates to raynet/build.sh)
# --------------------------------------------------------------------------- #
build_raynet() {
    ensure_symlinks
    log "RayNet: simlibs + omnetbind via build.sh"
    cd "$RAYNET_PATH"
    local flags=""
    [[ -d "$RAYNET_VENV_PATH" ]] || flags+=" -i"   # create venv if missing
    [[ "$REBUILD" == "true" || "$CLEAN_ALL" == "true" ]] && flags+=" -r"
    [[ "$CLEAN_ALL" == "true" ]] && flags+=" -c"
    # build.sh prompts if the venv python != 3.12; auto-confirm.
    yes y | ./build.sh $flags
    find "$RAYNET_PATH/build" -name '*.so' | grep -q . || die "omnetbind .so not produced"
}

# --------------------------------------------------------------------------- #
# Drive
# --------------------------------------------------------------------------- #
run_all() {
    build_omnet
    build_inet
    build_ext
    [[ -d "$RAYNET_VENV_PATH" && "$CLEAN_ALL" != "true" ]] || build_venv
    build_raynet
}

if [[ -n "$STAGE" ]]; then
    case "$STAGE" in
        omnet)  build_omnet  ;;
        inet)   build_inet   ;;
        ext)    build_ext    ;;
        venv)   build_venv   ;;
        raynet) build_raynet ;;
        *)      die "unknown stage: $STAGE (omnet|inet|ext|venv|raynet)" ;;
    esac
elif [[ "$REBUILD" == "true" ]]; then
    # Fast path: just rebuild RayNet against already-built OMNeT++/INET/extensions.
    build_raynet
else
    run_all
fi

log "DONE"

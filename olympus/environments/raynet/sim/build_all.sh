#!/usr/bin/env bash
#
# Build / rebuild the full RayNet + OMNeT++ simulation stack that is vendored
# under this directory as git submodules:
#
#   sim/omnetpp    OMNeT++ 6.x        (headless: Cmdenv only)
#   sim/inet4.5    INET 4.5 fork      -> available in omnetpp/samples/inet4.5
#   omnetpp/samples/{tcpPaced,cubic,...}
#                  INET extension repos used by the congestion-control sims
#   sim/raynet     RayNet simlibs + omnetbind pybind module + Python venv
#
# The script is self-locating: it derives all paths from its own location, so it
# works no matter where the Olympus repo is checked out. Build order encodes the
# dependency chain: omnetpp -> inet -> INET extensions -> raynet.
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
export OMNET_SAMPLES_PATH="$OMNET_PATH/samples"
export TCPPACED_PATH="$OMNET_PATH/samples/tcpPaced"
export CUBIC_PATH="$OMNET_PATH/samples/cubic"
export RAYNET_VENV_PATH=$RAYNET_PATH/.venv
export CCACHE_DIR="${CCACHE_DIR:-/tmp/olympus-ccache}"
mkdir -p "$CCACHE_DIR"

INET_EXTENSION_REPOS=(
    "tcpPaced:https://github.com/CJUKnowles/tcpPaced.git"
    "tcpGoodputApplications:https://github.com/Avian688/tcpGoodputApplications.git"
    "cubic:https://github.com/Avian688/cubic.git"
    "os3:https://github.com/Avian688/os3.git"
    "leosatellites:https://github.com/Avian688/leosatellites.git"
    "orbtcp:https://github.com/Avian688/orbtcp.git"
    "bbr:https://github.com/Avian688/bbr.git"
    "orbtcpExperiments:https://github.com/Avian688/orbtcpExperiments.git"
)

JOBS="$(nproc)"
STAGE=""
REBUILD="false"
CLEAN_ALL="false"

log() { echo -e "\n=== [build_all] $* ($(date '+%H:%M:%S')) ==="; }
die() { echo "ERROR: $*" >&2; exit 1; }
show_help() {
    cat <<'EOF'
Build / rebuild the full RayNet + OMNeT++ simulation stack.

Usage:
  ./build_all.sh                 full incremental build of the whole stack
  ./build_all.sh -r|--rebuild    clean-rebuild RayNet only
  ./build_all.sh -c|--clean-all  clean and rebuild EVERYTHING from scratch
  ./build_all.sh -s|--stage S    build a single stage:
                                 omnet | inet | ext | venv | raynet
  ./build_all.sh -j N            parallel jobs (default: nproc)
  ./build_all.sh -h|--help
EOF
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--rebuild)   REBUILD="true"; shift ;;
        -c|--clean-all) CLEAN_ALL="true"; shift ;;
        -s|--stage)     STAGE="${2:-}"; shift 2 ;;
        -j)             JOBS="${2:-}"; shift 2 ;;
        -h|--help)      show_help; exit 0 ;;
        *)              die "unknown option: $1 (try --help)" ;;
    esac
done

# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
[[ -d "$OMNET_PATH"  ]] || die "OMNeT++ submodule missing: $OMNET_PATH (run: git submodule update --init)"
[[ -d "$RAYNET_PATH" ]] || die "RayNet submodule missing: $RAYNET_PATH  (run: git submodule update --init)"
command -v python3.12 >/dev/null 2>&1 || die "python3.12 not found (RayNet requires it)"

# Ensure INET is visible under omnetpp/samples. The repo still vendors INET as a
# flat submodule, while OMNeT++ projects expect to discover it from samples/.
ensure_inet_sample() {
    local link="$INET_PATH"
    if [[ ! -e "$link" ]]; then
        log "restoring missing symlink samples/inet4.5 -> ../../inet4.5"
        ln -s "../../inet4.5" "$link"
    fi
}

extension_dir() {
    local name="$1"
    printf '%s/%s\n' "$OMNET_SAMPLES_PATH" "$name"
}

ensure_extension_repo() {
    local name="$1"
    local repo="$2"
    local dir
    local actual_repo
    dir="$(extension_dir "$name")"

    if [[ -L "$dir" ]]; then
        log "replacing samples/$name symlink with an in-place checkout"
        rm "$dir"
    elif [[ -e "$dir/.git" ]]; then
        actual_repo="$(git -C "$dir" remote get-url origin 2>/dev/null || true)"
        if [[ "$actual_repo" != "$repo" ]]; then
            if [[ "$CLEAN_ALL" == "true" ]]; then
                log "replacing samples/$name checkout: origin is $actual_repo, expected $repo"
                rm -rf "$dir"
            else
                die "$dir has origin $actual_repo, expected $repo (rerun with --clean-all to replace it)"
            fi
        else
            return 0
        fi
    elif [[ -e "$dir" && "$CLEAN_ALL" == "true" ]]; then
        log "removing non-git samples/$name before clean clone"
        rm -rf "$dir"
    elif [[ -e "$dir" ]]; then
        die "$dir already exists but is not a git checkout"
    fi

    log "cloning INET extension $name into $dir"
    git clone "$repo" "$dir"
}

ensure_inet_extensions() {
    [[ -d "$OMNET_SAMPLES_PATH" ]] || die "OMNeT++ samples directory missing: $OMNET_SAMPLES_PATH"

    local item
    local name
    local repo
    for item in "${INET_EXTENSION_REPOS[@]}"; do
        name="${item%%:*}"
        repo="${item#*:}"
        ensure_extension_repo "$name" "$repo"
    done
}

project_is_built() {
    local name="$1"
    local dir="$2"
    [[ -f "$dir/Makefile" ]] || return 1
    [[ -f "$dir/src/lib${name}.so" ]]
}

build_extension_project() {
    local name="$1"
    local dir
    dir="$(extension_dir "$name")"

    [[ -f "$dir/Makefile" ]] || {
        log "extension: $name has no top-level Makefile; skipping build"
        return 0
    }

    if [[ "$CLEAN_ALL" != "true" ]] && project_is_built "$name" "$dir"; then
        log "extension: $name already appears to be built"
        return 0
    fi

    log "extension: $name"
    cd "$dir"
    [[ "$CLEAN_ALL" == "true" ]] && make cleanall || true

    local makemake_args
    makemake_args=(-f --deep --make-so -o "$name" -O out)

    add_inet_dep() {
        makemake_args+=(
            -KINET_PROJ="$INET_PATH"
            -DINET_IMPORT
            -I"$INET_PATH/src"
            -L"$INET_PATH/src"
            -lINET
        )
    }

    add_extension_dep() {
        local dep="$1"
        local dep_dir
        dep_dir="$(extension_dir "$dep")"
        makemake_args+=(
            -I"$dep_dir/src"
            -L"$dep_dir/src"
            -l"$dep"
        )
    }

    add_pkg_config_flags() {
        local package="$1"
        local flags
        if flags="$(pkg-config --cflags --libs "$package" 2>/dev/null)"; then
            # shellcheck disable=SC2206
            makemake_args+=($flags)
        fi
    }

    add_include_dir_if_present() {
        local include_dir="$1"
        [[ -d "$include_dir" ]] && makemake_args+=(-I"$include_dir")
        return 0
    }

    case "$name" in
        tcpPaced|tcpGoodputApplications)
            add_inet_dep
            ;;
        cubic)
            add_inet_dep
            add_extension_dep tcpPaced
            ;;
        os3)
            add_inet_dep
            add_pkg_config_flags libcurl
            add_include_dir_if_present /usr/include/curl
            add_include_dir_if_present /usr/include/x86_64-linux-gnu/curl
            ;;
        leosatellites)
            add_inet_dep
            add_extension_dep os3
            add_pkg_config_flags igraph
            ;;
        orbtcp)
            add_inet_dep
            add_extension_dep tcpPaced
            ;;
        bbr)
            add_inet_dep
            add_extension_dep tcpGoodputApplications
            add_extension_dep tcpPaced
            ;;
        orbtcpExperiments)
            add_inet_dep
            add_extension_dep bbr
            add_extension_dep cubic
            add_extension_dep leosatellites
            add_extension_dep orbtcp
            add_extension_dep os3
            add_extension_dep tcpGoodputApplications
            add_extension_dep tcpPaced
            ;;
        *)
            add_inet_dep
            ;;
    esac

    find src \( -name '*_m.cc' -o -name '*_m.h' \) -delete
    rm -rf out
    ( cd src && opp_makemake "${makemake_args[@]}" )
    make -j"$JOBS" MODE=release
    [[ -f "src/lib${name}.so" ]] || die "lib${name}.so not produced"
}

write_raynet_build_paths() {
    log "RayNet: writing build_paths.sh for this checkout"
    cat > "$RAYNET_PATH/build_paths.sh" <<'EOF'
# ! SET YOUR ENVIRONMENT VARIABLES HERE ! #
export RAYNET_PATH=$HOME/olympus/olympus/environments/raynet/sim/raynet
export OMNET_PATH=$HOME/olympus/olympus/environments/raynet/sim/omnetpp
export INET_PATH=$HOME/olympus/olympus/environments/raynet/sim/inet4.5

export RAYNET_VENV_PATH=$RAYNET_PATH/.venv
# ! BUILD WILL FAIL IF THESE ARE INCORRECT ! #
EOF
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

    if [[ "$CLEAN_ALL" == "true" || ! -f .venv/bin/activate ]]; then
        log "OMNeT++: creating python virtual environment in .venv"
        python3.12 -m venv .venv --upgrade-deps --clear --prompt "$(basename "$OMNET_PATH")/.venv"
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    python3 -m pip install -r python/requirements.txt

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
    ensure_inet_sample
    log "INET: install Python requirements + make makefiles + make -j$JOBS MODE=release"
    set +u; source "$OMNET_PATH/setenv"; set -u
    cd "$INET_PATH"
    set +u; source ./setenv; set -u
    [[ "$CLEAN_ALL" == "true" ]] && make clean MODE=release || true
    python3 -m pip install -r python/requirements.txt
    make makefiles
    make -j"$JOBS" MODE=release
    find . -name 'libINET*.so' -print -quit | grep -q . || die "libINET not produced"
}

# --------------------------------------------------------------------------- #
# Stage: external extensions (clones/builds the same repos as cc_install_scripts)
# --------------------------------------------------------------------------- #
build_ext() {
    ensure_inet_sample
    ensure_inet_extensions
    set +u; source "$OMNET_PATH/setenv"; set -u

    local item
    local name
    for item in "${INET_EXTENSION_REPOS[@]}"; do
        name="${item%%:*}"
        build_extension_project "$name"
    done
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
    ensure_inet_sample
    write_raynet_build_paths
    log "RayNet: simlibs + omnetbind via build.sh"
    cd "$RAYNET_PATH"
    # A relocated checkout leaves a stale CMakeCache pointing at the old path,
    # which makes cmake refuse to reconfigure. Drop it so omnetbind rebuilds cleanly.
    if [[ -f build/CMakeCache.txt ]] && ! grep -q "$RAYNET_PATH/build" build/CMakeCache.txt; then
        log "removing stale build/ (CMakeCache from a different path)"
        rm -rf build
    fi
    local flags=""
    [[ -d "$RAYNET_VENV_PATH" ]] || flags+=" -i"   # create venv if missing
    [[ "$REBUILD" == "true" || "$CLEAN_ALL" == "true" ]] && flags+=" -r"
    [[ "$CLEAN_ALL" == "true" ]] && flags+=" -c"
    # build.sh prompts if the venv python != 3.12; auto-confirm via process
    # substitution (a plain `yes |` pipe trips pipefail with SIGPIPE 141).
    ./build.sh $flags < <(yes y)
    find "$RAYNET_PATH/build" -name '*.so' | grep -q . || die "omnetbind .so not produced"
}

# --------------------------------------------------------------------------- #
# Drive
# --------------------------------------------------------------------------- #
run_all() {
    build_omnet
    build_inet
    build_ext
    # Always (re)create the venv on a full build: a relocated venv has stale
    # absolute paths baked in and cannot be reused.
    build_venv
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

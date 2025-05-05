# Set project root directory to use absolute paths below.
export PRJ_ROOT_DIR=""
if [ "$GITHUB_ACTIONS" = "true" ]; then
    PRJ_ROOT_DIR="$GITHUB_WORKSPACE"
else
    if [ -n "${BASH_SOURCE[0]:-}" ]; then
        # shellcheck disable=SC3000-SC4000
        PRJ_ROOT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
    else
        PRJ_ROOT_DIR="$(dirname "$(realpath "$0")")"
    fi
fi

# Set path to implementations
export OIF_IMPL_PATH="${PRJ_ROOT_DIR}/oif_impl/oif/_impl:${PRJ_ROOT_DIR}/lang_python/oif_impl/openinterfaces/_impl"

# Add compiled libraries to the path for the linker.
if [ -n "$LD_LIBRARY_PATH" ]; then
    export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
fi

export LD_LIBRARY_PATH="$PRJ_ROOT_DIR/build":"$LD_LIBRARY_PATH"

# Find Python packages inside the `src` directory.
if [ -n "$PYTHONPATH" ]; then
    export _OLD_PYTHONPATH="$PYTHONPATH"
fi

oifpath=${PRJ_ROOT_DIR}/oif/interfaces/python
oifpath="$oifpath":${PRJ_ROOT_DIR}/lang_python/oif
oifpath="$oifpath":${PRJ_ROOT_DIR}/lang_python/oif_interfaces
oifpath="$oifpath":${PRJ_ROOT_DIR}/lang_python/oif_impl
oifpath="$oifpath":${PRJ_ROOT_DIR}/build
export PYTHONPATH="$oifpath":"$PYTHONPATH"

export JULIA_PROJECT="$PRJ_ROOT_DIR"

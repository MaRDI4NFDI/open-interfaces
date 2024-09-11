# Set project root directory to use absolute paths below.
if [ "$GITHUB_ACTIONS" = "true" ]; then
    export PRJ_ROOT_DIR="$GITHUB_WORKSPACE"
else
    export PRJ_ROOT_DIR="$(dirname $(realpath "$0"))"
fi

# Set path to implementations
export OIF_IMPL_ROOT_DIR="${PRJ_ROOT_DIR}"

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
oifpath="$oifpath":${PRJ_ROOT_DIR}/oif/lang_python
oifpath="$oifpath":${PRJ_ROOT_DIR}/oif_impl/lang_python
oifpath="$oifpath":${PRJ_ROOT_DIR}/oif_impl/impl
oifpath="$oifpath":${PRJ_ROOT_DIR}/build
export PYTHONPATH="$oifpath":"$PYTHONPATH"

export JULIA_PROJECT="$PRJ_ROOT_DIR"

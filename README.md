# C to multiple experiments

## Setup

Only tested setup is based on the shipped `Dockerfile`.
That means debian bullseye with julia 1.7.

### Container based
#### CLion
Using this with CLion means setting up a new docker toolchain
with the base image from the `Dockerfile` and then selecting
that toolchain when CLion configures cmake.
[See also](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html)

#### VSCode (not tested)
Should be a similar setup to CLion, using the "Docker Tools" extension.

### Local host setup

See the [container base image](https://zivgitlab.uni-muenster.de/ag-ohlberger/mardi/container/-/blob/main/m2-dev/Dockerfile)
for a list of necessary (debian) packages to install.

Then run
```shell
BUILDDIR=/some/path
mkdir ${BUILDDIR}
cd ${BUILDDIR}
cmake ..
cmake --build .
```

## Execution

If you want to run from the build dir
setting `LD_LIBRARY_PATH` is necessary
```shell
cd ${BUILDDIR}
export LD_LIBRARY_PATH=${BUILDDIR}/lang_julia:${BUILDDIR}/connector:${LD_LIBRARY_PATH}
./lang_c/oif_c
./lang_python/oif_python.py
```

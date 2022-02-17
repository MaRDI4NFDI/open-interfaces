# C to multiple experiments

## Setup

Only tested setup is based on the shipped `Dockerfile`.
That means debian bullseye with julia 1.7.

### CLion
Using this with CLion means setting up a new docker toolchain
with the base image from the `Dockerfile` and then selecting 
that toolchain when CLion configures cmake.
[See also](https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html)

### VSCode (not tested)
Should be a similar setup to CLion, using the "Docker Tools" extension.
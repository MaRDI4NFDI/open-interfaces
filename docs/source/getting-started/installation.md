# Installation

Use [`conda`](
    https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments
) or [`mamba`](
https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba-user-guide
) package manager to create the development environment from provided
environment files:
```shell
conda env create -n env-name -f environment-linux.yaml
```
if you use Linux or
```shell
conda env create -n env-name -f environment-macos.yaml
```

## Build

To build the software, use command
```shell
    make
```
which invokes underlying CMake build and builds software inside
the `build` directory.


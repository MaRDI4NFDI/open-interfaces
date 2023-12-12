# Description

This repository contains toy example that I am working on to understand
how to develop `open-interface` project within MaRDI initiative.

This toy example investigates how to connect users using Python with software
written in C and Python using a broker library in-between.


## Installation for development

Use `conda` or `mamba` package manager:
```shell
mamba env create -n env-name -f environment-linux.yaml
```
if you use Linux or
```shell
mamba env create -n env-name -f environment-macos.yaml
```

## Build

To build the software, use command
```shell
    make
```
which invokes underlying CMake build and builds software inside
the `build` directory.

## Quality assurance during development

To ensure code consistency, we use [`pre-commit`](https://pre-commit.com/).
It is configured to run multiple checks for formatting and trailing whitespace
for all source code in the repo.
During development, the checks can be run automatically by installing
a pre-commit Git hook:

    pre-commit install

or by invoking it manually via

    pre-commit run --all-files

We recommend running it automatically so that the code is pushed only after
formatting checks.

## Run examples

One needs set `PYTHONPATH`:
```shell
export PYTHONPATH=src/oif/backend_python/src:
```

### Run examples from Python

To run examples on invoking quadratic solver from Python, run
```shell
python examples/call_qeq_from_python.py [c|python]
```
where backend specification is optional and defaults to C backend.

### Run examples from C

To run examples on invoking quadratic solver from Python, run
```shell
build/examples/call_qeq_from_c [c|python]
```
where backend specification is optional and defaults to C backend.

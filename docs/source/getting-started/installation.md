# Installation

This page describes how to install and build Open Interfaces
on a Unix-like system, such as Ubuntu or macOS.
We do not support Windows at the moment, unfortunately,
but it might be possible to install Open Interfaces
using Windows Subsystem for Linux (WSL).

## Overview and prerequisites

The installation of Open Interfaces depends on what programming languages
you want to use (bindings and implementations) as well as the preferred
package manager for installing dependencies.

We assume that the following software is installed on your system:
- `curl` or `git` for downloading the source code
- Package manager such as (`apt`)[https://wiki.debian.org/apt] for Debian-like
  systems,
  (`brew`)[https://brew.sh/] for macOS and Linux, or (`conda`)[https://conda.io]/
  (`mamba`)[https://mamba.readthedocs.io/en/latest/],
  for installing dependencies packages.

Then you can choose one of the following options:
- If you interested only in Python bindings and implementations,
  you can install them via `pip` as described
  [here](#install-via-pip), which builds Open Interfaces automatically.
- If you interested in C and Python bindings
  and implementations, you can install dependencies via `conda`
  as described [here](#install-via-conda) and then proceed with building the
  source code.
- If you want to use Julia and Julia packages,
  please follow instructions [here](#install-julia).

## The source code

Download the archive with the source code of the
(latest release of Open
Interfaces)[https://github.com/MaRDI4NFDI/open-interfaces/releases/latest]
and unpack the content to a directory of your choice.
Then open a terminal and go to that directory.

Alternatively, download and unpack the source code directly from the terminal,
for example, for the latest release:
```shell
curl -LO https://github.com/MaRDI4NFDI/open-interfaces/archive/refs/tags/v0.5.5.tar.gz
tar -xzvf v0.5.5.tar.gz
cd open-interfaces-0.5.5
```

If you prefer to use the latest development version,
you can obtain it by cloning the repository via `git`:
```shell
git clone git@github.com:MaRDI4NFDI/open-interfaces.git      # via SSH
# or
git clone https://github.com/MaRDI4NFDI/open-interfaces.git  # via HTTPS
```
and then change to the repository directory:
```shell
cd open-interfaces
```

## Dependencies

Open Interfaces depends on several external libraries and tools,
such as CMake, a C compiler, Python interpreter, Julia, SUNDIALS,
SciPy, etc.

All dependencies can be installed through a package manager,
such as `apt`, `brew`, or `conda`.

### Mandatory dependencies

The following dependencies are mandatory for building and running Open
Interfaces:
- A C compiler supporting the C17 standard, such as
    [`gcc`](https://https://gcc.gnu.org/) 9+ or
    [`clang`](https://clang.llvm.org/) 7+
- The [CMake](https://cmake.org) meta-build system v3.18+
- [`libffi`](https://sourceware.org/libffi/) v8+
- The [Make](https://www.gnu.org/software/make/) build system v4+

### Optional recommended dependencies

The following dependencies are optional but recommended:

To really benefit from Open Interfaces, it is recommended to install
C implementations and Python and Julia interfaces and implementations:

- SUNDIALS v6.0+
- Python 3.9 with NumPy, SciPy, and Matplotlib packages
- Julia 1.10+

To simplify the installation of dependencies,
C implementations and Python interface and packages can be installed via
`conda` using provided environment files as described
[here](#install-via-conda).

Alternatively, if your preferred programming language is Python,
you can install Open Interfaces via `pip` as described
[here](#install-via-pip), which builds Open Interfaces automatically,
so you do not have to build the library by hand.

For instructions on how to install Julia and Julia packages,
please go [here](#install-julia).

(install-via-conda)=
### Option 1: Installing dependencies via `conda`

Using the [`conda`][1] package manager helps with obtaining software dependencies
such as C compiler, Python interpreter, SUNDIALS, etc.
Additionally, all dependencies (except Julia and Julia packages)
will be installed in a separate environment,
therefore, there will be no conflicts with system packages.

If `conda` is not installed, it can be installed via the following command:
```shell
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

After that, create an environment for Open Interfaces and install the dependencies
using the provided `environment-linux.yaml` file:
```shell
conda env create -n open-interfaces -f environment-linux.yaml
```

Then activate the created environment:
```shell
conda activate open-interfaces
```

Note that the environment name `open-interfaces` can be changed at your will.

(install-via-pip)=
### Option 2: Installation of Python bindings and implementations

The Python bindings and implementations of the interfaces are available
from [Python Package Index (PyPI)](https://pypi.org/)
and can be installed using
```shell
pip install openinterfaces
```

(install-julia)=
### Installing Julia and Julia packages

Then Julia must be installed via the following command
from the [official instructions][2]:
```shell
curl -fsSL https://install.julialang.org | sh
```

Then we can activate the environment for Open Interfaces
and install required Julia packages:
```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Building the source code

After the necessary dependencies are installed,
and environments are activated,
we can build Open Interfaces.

To build the software, use command
```shell
    make release
```
which invokes underlying CMake build and builds Open Interfaces inside
the `build` directory.

## Activating the environment

Finally, we set auxiliary environment variables via the following command:
```shell
source env.sh
```
that sets `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`, and `OIF_IMPL_PATH`
variables.
These variables are necessary for running examples and tests.

## Checking the build

To check that the build was successful,
you can run the provided tests via command
```shell
    make test
```
which should take no more than a few minutes to complete.

## Running examples

Open Interfaces comes with several examples
the source code of which can be found in the `examples` directory
of the source code.

For guidance on how to run them,
please refer to the specific pages in the Table of Contents.

## Building the source code for development

If you plan to work on the source code,
it is recommended to build it in the `Debug` mode with the command
```shell
    make
```
which enables additional runtime checks and debugging symbols,
particularly useful when using a debugger such as `gdb` or `lldb`.

Also, there is a special configuration for development that
helps with finding memory leaks and memory corruption errors:
```shell
    make debug-verbose-info-and-sanitize
```
which should be used when working on the C source code mostly,
as sanitizers are not compatible with Python and Julia bindings,
as well as due to the verbosity of debugging messages.

### Quality assurance during development

To ensure code consistency,
we use [`pre-commit`](https://pre-commit.com/).
It is configured to run multiple checks for formatting and trailing whitespace
for all source code in the repository.
During development, the checks can be run automatically by installing
a pre-commit Git hook:

    pre-commit install

or by invoking it manually via

    pre-commit run --all-files

We recommend running it automatically so that the code is pushed only after
running these checks to avoid wasting time and computational resources
on the continuous-integration service.


[1]: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
[2]: https://julialang.org/downloads/

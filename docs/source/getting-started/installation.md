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
- Package manager such as [`apt`](https://wiki.debian.org/apt) for Debian-like
  systems,
  [`brew`](https://brew.sh/) for macOS and Linux, or [`conda`](https://conda.io)/
  [`mamba`](https://mamba.readthedocs.io/en/latest/) for Linux/macOS/Windows,
  for installing the dependencies.

Then you can choose one of the following options:
- If you interested in C and Python bindings
  and all implementations, you can install dependencies via `conda`
  as described [here](#install-via-conda) (this is the recommended option)
  and then proceed with [building the source code](#building-the-source-code).
- If you interested only in Python bindings,
  you can install them via `pip` as described
  [here](#install-via-pip), which builds Open Interfaces automatically.
- If you want to use Julia and Julia packages,
  please follow instructions [here](#install-julia).

(building-the-source-code)=
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

Open Interfaces depend on several external libraries and tools,
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
- A C++ compiler supporting the C++17 standard, such as
    [`g++`](https://https://gcc.gnu.org/) 9+ or
    [`clang++`](https://clang.llvm.org/) 7+
- The [CMake](https://cmake.org) meta-build system v3.18+
- [`libffi`](https://sourceware.org/libffi/) v8+
- The [Make](https://www.gnu.org/software/make/) build system v4.3+
- [`pkg-config`](https://www.freedesktop.org/wiki/Software/pkg-config/) v0.29+

On Ubuntu, these dependencies can be installed via:
```shell
sudo apt install build-essential cmake libffi-dev pkg-config
```

With these dependencies installed, very minimal build can be done
enabling C bindings and one (test-level) implementation in C.
It is recommended to follow with the next section before building the source
code.

### Optional recommended dependencies

The following dependencies are optional but recommended:

To really benefit from _Open Interfaces_, it is recommended
to install additional dependencies which enable more implementations
(and also languages).

For C implementations LAPACK library and SUNDIALS v6.0+
will enable `c_lapack` implementation of the `linsolve` interface
and `sundials_cvode` implementation of the `ivp` interface, respectively.
On Ubuntu, one can install them via:
```shell
sudo apt install liblapack-dev libsundials-dev
```

For Python bindings and implementations Python 3.9+ with header files
along with the NumPy, SciPy, and Matplotlib packages.
On Ubuntu, one can install them via the system package manager:
```shell
sudo apt install python3-dev python3-numpy python3-scipy python3-matplotlib
```

For Julia bindings and implementations, Julia 1.10+ is required
along with several Julia packages. The only supported installation method
for Julia and Julia packages is described [in this section](#install-julia).

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
### Option 2: Installation via `pip`

If you are interested mostly in Python bindings as well as
C and Python implementations,
then the easiest way to install Open Interfaces
is via `pip`
from the [Python Package Index (PyPI)](https://pypi.org/).

First, ensure that essential dependencies for Python are installed.
On Ubuntu, this can be done via (in addition to the mandatory dependencies):
```shell
sudo apt install python3-dev python3-venv python3-numpy python3-scipy python3-msgpack
```
Alternatively, you can install them via `conda` as described
[in the previous section](#install-via-conda).

Optional dependencies include `LAPACK` and `SUNDIALS` libraries
for C implementations, which can be installed via:
```shell
sudo apt install liblapack-dev libsundials-dev
```

Then it is recommended to create a virtual environment for Python:
```shell
python3 -m venv .venv
```
and activate it:
```shell
source .venv/bin/activate
```

Finally, install Open Interfaces via `pip`:
```shell
pip install openinterfaces
```

(install-julia)=
### Installing Julia and Julia packages

If you are interested in using Julia bindings and implementations,
you need to install Julia and required Julia packages.
Currently, Julia packages are not available, but the instructions
in this section will help you to set up the environment
and install Julia dependencies via provided project files.

The [recommend way of installing Julia](https://julialang.org/install/)
is via the following command:
```shell
curl -fsSL https://install.julialang.org | sh
```

Then, assuming that the current working directory is the source code
of Open Interfaces,
create and activate a Julia environment for Open Interfaces
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
    make
```
which invokes `cmake` and builds Open Interfaces inside
the `build` subdirectory enabling compiler optimizations.

## Activating the environment

Finally, we set auxiliary environment variables via the following command:
```shell
source env.sh
```
that sets `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`, and `OIF_IMPL_PATH`
variables.
These variables are necessary for running examples and tests.
The last variable, `OIF_IMPL_PATH`, is used to point to the directories
where implementations are searched for.

## Checking the build

To check that the build was successful,
you can run the provided tests via command
```shell
    make test
```
which should take no more than a few minutes to complete.
In case when not all dependencies are installed,
some tests might be skipped if the corresponding implementations
or languages are not available.

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
    make debug
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

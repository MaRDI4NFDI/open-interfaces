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
  systems or [`conda`](https://conda.io)/
  [`mamba`](https://mamba.readthedocs.io/en/latest/) for Linux,
  for installing the dependencies.
- Xcode Command Line Tools and package manager [`brew`](https://brew.sh/) for macOS
  (this is currently the only supported option)

If the prerequisites are installed:
- [Obtain the source code](#source-code)
- Read about mandatory and optional dependencies in the [Dependencies
    section](#dependencies)
- Install dependencies:
  * [On macOS](#install-macos)
  * [On Ubuntu](#install-ubuntu)
- [Build the source](#build-source-code)
- [Run tests](#run-tests)
- [Run examples](#run-examples)


(source-code)=
## The source code

Download the archive with the source code of the
(latest release of Open
Interfaces)[https://github.com/MaRDI4NFDI/open-interfaces/releases/latest]
and unpack the content to a directory of your choice.
Then open a terminal and go to that directory.

Alternatively, download and unpack the source code directly from the terminal,
for example, for the latest release:
```shell
curl -LO https://github.com/MaRDI4NFDI/open-interfaces/archive/refs/tags/v0.6.0.tar.gz
tar -xzvf v0.6.0.tar.gz
cd open-interfaces-0.6.0
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

(dependencies)=
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

For Julia bindings and implementations, Julia 1.10+ is required
along with several Julia packages. The only supported installation method
for Julia and Julia packages is described [in this section](#install-julia).

If your preferred programming language is Python,
you can install Open Interfaces via `pip` as described
[here](#install-via-pip), which builds Open Interfaces automatically,
so you do not have to build the library by hand.

(install-ubuntu)=
## Installation on Ubuntu

Mandatory and optional dependencies can be installed on Ubuntu
using the `apt` package manager as described
[in this section](#install-ubuntu-via-apt)
or alternatively via `conda` as described [here](#install-ubuntu-via-conda).

(install-ubuntu-via-apt)=
### Installation on Ubuntu via `apt`
- Mandatory dependencies:
  ```shell
  sudo apt install build-essential cmake libffi-dev pkg-config
  ```

- Python development headers, scientific packages and test framework:
  ```shell
  sudo apt install python3-dev python3-numpy python3-scipy python3-matplotlib python3-pytest
  ```
- For instructions on how to install Julia and Julia packages,
  please go [here](#install-julia).

(install-ubuntu-via-conda)=
### Installation on Ubuntu via `conda`

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
and proceed with [building the source code](#build-source-code).


(install-macos)=
## Installation on macOS

On modern macOS systems with Apple Silicon architecture the only supported
installation depends on Apple-provided core development tools and additional
dependencies such as Python or OpenBlAS installed via the Homebrew package manager.

1. Install development tools (Clang, Clang++, Make, etc.):
   ```shell
     xcode-select install
   ```

2. Install CMake via Homebrew:
   ```
   brew install cmake
   ```

3. (Optional) Install C implementations:
   ```shell
   brew install openblas sundials
   ```

4. (Optional) Install Python:
   ```shell
   brew install python
   ```

5. (Optional) Create Python virtual environment and install Python packages (recommended):
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-macos.txt
   ```

6. (Optional) Install Julia
Instructuctions are [here](#install-julia).

7. Proceed with [building the source code](#build-source-code).


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
julia --project=. -e 'using Pkg;
  Pkg.add(["Libdl", "MsgPack", "OrdinaryDiffEq", "NonlinearSolve"]);
  Pkg.develop(path="lang_julia/OpenInterfaces");
  Pkg.develop(path="lang_julia/OpenInterfacesImpl");
  Pkg.instantiate()'
```

(build-source-code)=
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

(run-tests)=
## Checking the build

We set auxiliary environment variables via the following command:
```shell
source env.sh
```
that sets `PATH`, `LD_LIBRARY_PATH`, and `PYTHONPATH`, and `OIF_IMPL_PATH`
variables.
These variables are necessary for running examples and tests.
The last variable, `OIF_IMPL_PATH`, is used to point to the directories
where implementations are searched for.

To check that the build was successful,
you can run the provided tests via command
```shell
    make test
```
which should take no more than a few minutes to complete.
In case when not all dependencies are installed,
some tests might be skipped if the corresponding implementations
or languages are not available.

(run-examples)=
## Running examples

Open Interfaces comes with several examples
the source code of which can be found in the `examples` directory
of the source code.

For guidance on how to run them,
please refer to the specific pages in the Table of Contents.

(install-via-pip)=
## Installation of Python bindings via `pip`

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

## For developers: Building the source code for development

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

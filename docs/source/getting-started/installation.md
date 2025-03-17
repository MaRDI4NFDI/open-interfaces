# Installation

## Prerequisites

Prerequisites are `curl`, `git` and `make`.
If they're not installed, use the following command to install them:
```shell
sudo apt install curl git make
```

## Installation via `conda`

Installation via [`conda`][1] helps with obtaining software dependencies
such as C compiler, Python interpreter, SUNDIALS, etc.
Additionally, all dependencies (except Julia and Julia packages)
will be installed in a separate environment,
therefore, there will be no conflicts with system packages.

If `conda` is not installed, it can be installed via the following command:
```shell
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Then clone the Open Interfaces repository and go to the repository directory:
```shell
git clone https://github.com/MaRDI4NFDI/open-interfaces
cd open-interfaces
```

After that, create an environment for Open Interfaces and install dependencies:
```shell
conda env create -n open-interfaces -f environment-linux.yaml
```

Then activate the created environment:
```shell
conda activate open-interfaces
```

Note that the environment name `open-interfaces` can be changed at your will.

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

Finally, we set auxiliary environment variables via the following command:
```shell
source env.sh
```

## Build

After the necessary dependencies are installed,
and environments are activated,
we can build Open Interfaces.

To build the software, use command
```shell
    make release
```
which invokes underlying CMake build and builds Open Interfaces inside
the `build` directory.

To test that the build processes has succeeded, use command
```shell
    make test
```


[1]: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
[2]: https://julialang.org/downloads/

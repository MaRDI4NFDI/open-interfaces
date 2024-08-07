# MaRDI Open Interfaces

_MaRDI Open Interfaces_ is a project aiming to improve interoperability
in scientific computing by removing two hurdles that computational scientists
usually face in their daily work.

These hurdles are the following.
First, numerical solvers are often implemented in different programming
languages.
Second, these solvers have potentially significantly diverging interfaces
in terms of function names, order of function arguments, and the invocation
order.
Therefore, when a computational scientist wants to switch from one solver
to another, it could take non-negligible effort in code modification
and testing for the correctness.

_Open Interfaces_ aim to alleviate these problems by providing automatic data
marshalling between different languages and a set of interfaces for typical
numerical problems such as integration of differential equations and
optimization.

This project is the part of the [_Mathematical Research Data Initiative
(MaRDI)_](https://mardi4nfdi.de).

## Data flow

![Architecture of the MaRDI Open Interfaces](assets/arch.png)

This figure shows the software architecture of the _MaRDI Open Interfaces_.
There are two principal decoupled parts. The left part is user-facing
and allows a user to request an implementation of some numerical procedure
and then invoke different functions in this implementation to conduct
computations using a unified interface (Gateway)
that hides discrepancies between different implementations.
The other part (on the right) is completely hidden from the user
and works with an implementation of the interface.
Particularly, it loads the implementation and its adapter and converts
user data to the native data for the implementation.

## Installation for development

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

## Quality assurance during development

For quality assurance, we write unit tests that test communication between
different clients and solvers.
The full test suite can be run using the command
```shell
    make test
```

Additionally, to ensure code consistency,
we use [`pre-commit`](https://pre-commit.com/).
It is configured to run multiple checks for formatting and trailing whitespace
for all source code in the repository.
During development, the checks can be run automatically by installing
a pre-commit Git hook:

    pre-commit install

or by invoking it manually via

    pre-commit run --all-files

We recommend running it automatically so that the code is pushed only after
formatting checks.

## Run examples

Currently, running Open Interfaces requires setting several environment
variables to make sure that all necessary components can be found:
particularly compiled C libraries, Python and Julia modules, and
implementations.
To make it easier, a script is provided that sets all necessary variables,
and it must be sourced in the current shell:
```shell
source env.sh
```

### Run examples from Python

Let's try using _Open Interfaces_ by solving the Van der Pol equation:
```math
  \frac{\mathrm d^2 x}{\mathrm d t^2} - \mu
  \left(
    1 - x^{2}
  \right) \frac{\mathrm d x}{\mathrm d t} + x = 0, \quad
  x(0) = 2
```
with $\mu = 1000$
using different implementations of the IVP interface (interface for solving
initial-value problems for ordinary differential equations):
```shell
python examples/call_qeq_from_python.py [scipy_ode|sundials_cvode|jl_diffeq]
```
where the implementation argument is optional and defaults to `scipy_ode`.

This script uses stiff solvers for initial-value problems, why the value
of the parameter $\mu$ makes the system stiff.
At the end of the computations, the resultant solution is displayed.

### Run examples from C

Let's solve inviscid Burgers' equation:
```math
  \begin{aligned}
    &\frac{\partial u}{\partial t} +
        \frac{\partial \left( u^{2} / 2 \right)}{\partial x} = 0,
    \quad t \in [0, 2], \enspace x \in [0, 2] \\
    &u(t, 0) = 0.5 - 0.25 \sin \left( \pi x \right)\\
    &u(t, 0) = u(t, 2)
  \end{aligned}
```
from C. Run the following command
```shell
build/examples/call_ivp_from_c_burgers_eq [scipy_ode|sundials_cvode|jl_diffeq]
```
where the implementation argument is optional and defaults to `scipy_ode`.

The resultant solution is written to the file `solution.txt` and can be plotted
using any plotting software.


## Funding

This work is funded by the _Deutsche Forschungsgemeinschaft_ (DFG, _German
Research Foundation_) under Germany's Excellence Strategy EXC 2044-390685587,
“Mathematics Münster: Dynamics–Geometry–Structure” and the National Research
Data Infrastructure, project number&nbsp;460135501, NFDI&nbsp;29/1
“MaRDI – Mathematical Research Data Initiative
[Mathematische Forschungsdateninitiative]”.

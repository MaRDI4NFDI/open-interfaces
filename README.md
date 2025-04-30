<h1 align="center">
<img src="https://raw.githubusercontent.com/MaRDI4NFDI/open-interfaces/refs/heads/main/assets/mardi-oif-logo.svg" width="40" style="vertical-align: text-bottom;" />
MaRDI Open Interfaces
</h1>

[![QA](https://github.com/MaRDI4NFDI/open-interfaces/actions/workflows/qa.yaml/badge.svg)](https://github.com/MaRDI4NFDI/open-interfaces/actions/workflows/qa.yaml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13753666.svg)](https://doi.org/10.5281/zenodo.13753666)

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

![Architecture of the MaRDI Open Interfaces](https://media.githubusercontent.com/media/MaRDI4NFDI/open-interfaces/refs/heads/main/assets/arch.png)

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

## Installation of Python bindings and implementations

The Python bindings and implementations of the interfaces are available
from [Python Package Index (PyPI)](https://pypi.org/)
and can be installed using
```shell
pip install openinterfaces
```

## Run examples

Examples are provided in the `examples` directory in this repository.
Documentation explaining some of these examples is available here:
<https://mardi4nfdi.github.io/open-interfaces/>.


## Funding

This work is funded by the _Deutsche Forschungsgemeinschaft_ (DFG, _German
Research Foundation_) under Germany's Excellence Strategy EXC 2044-390685587,
“Mathematics Münster: Dynamics–Geometry–Structure” and the National Research
Data Infrastructure, project number&nbsp;460135501, NFDI&nbsp;29/1
“MaRDI – Mathematical Research Data Initiative
[Mathematische Forschungsdateninitiative]”.

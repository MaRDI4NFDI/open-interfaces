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

<p align="center">
    <img src="https://media.githubusercontent.com/media/MaRDI4NFDI/open-interfaces/refs/heads/main/assets/pairwise_bindings.png" />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://media.githubusercontent.com/media/MaRDI4NFDI/open-interfaces/refs/heads/main/assets/oif_bindings.png" />
    <br />
    <strong>A</strong>&nbsp;Traditional pairwise bindings
    <br />
    <strong>B</strong>&nbsp;Decoupled bindings via <em>Open Interfaces</em>
</p>

This project is the part of the [_Mathematical Research Data Initiative
(MaRDI)_](https://mardi4nfdi.de).

## Installation and Documentation

Please refer to the documentation at
<https://mardi4nfdi.github.io/open-interfaces/>
for deeper view on the goals and implementation details
as well as installation instructions, tutorials, and API reference.

## Support and Contact

_MaRDI Open Interfaces_ is an open source academic project.
Please use the
[issue tracker](https://github.com/MaRDI4NFDI/open-interfaces/issues)
for bug reports and feature requests and asking questions.


## License

[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)


## Citing

If you _MaRDI Open Interfaces_, please consider citing
[this paper](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.569):
```bibtex
@Article{KabanovEtAl2025,
  author    = {Kabanov, Dmitry I. and Rave, Stephan and Ohlberger, Mario},
  journal   = {Journal of Open Research Software},
  title     = {Improving Interoperability in Scientific Computing via {MaRDI Open Interfaces}},
  year      = {2025},
  issn      = {2049-9647},
  volume    = {13},
  doi       = {10.5334/jors.569},
  publisher = {Ubiquity Press, Ltd.},
}
```


## Funding

This work is funded by the _Deutsche Forschungsgemeinschaft_ (DFG, _German
Research Foundation_) under Germany's Excellence Strategy EXC 2044-390685587,
“Mathematics Münster: Dynamics–Geometry–Structure” and the National Research
Data Infrastructure, project number&nbsp;460135501, NFDI&nbsp;29/1
“MaRDI – Mathematical Research Data Initiative
[Mathematische Forschungsdateninitiative]”.

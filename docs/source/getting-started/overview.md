# Overview

This page describes high-level architecture and the goals of the project.

## Goals

This project is Measure 2 "Open Interfaces"
of the Mathematical Research Data Initiative (MaRDI) project.
Its goal is to provide interoperability between different languages
for scientific computing, such that users of one language could use
algorithm implementations from another language.

For example, a user can be mostly comfortable with Python, but desires
to use an implementation of a solver for partial differential equations
written in C.
If the solver doesn't offer Python bindings, the user needs to write
them with tools like `ctypes`, Python C API, Cython, etc.

If one needs to support $L$ languages and $I$ implementations,
that is, have bindings for multiple languages,
then the number of connections will be like shown on
{numref}`Figure %sA<overview-pairwise-vs-oif-bindings>`.

By introducing _MaRDI Open Interfaces_ --- a mediator library
that provides generic interfaces for common problems
in computational mathematics,
such as optimization or integration of differential equations,
and automates data marshalling between programming languages ---
then the number of connections is reduced as shown on
{numref}`Figure %sB<overview-pairwise-vs-oif-bindings>`.

(overview-pairwise-vs-oif-bindings)=
```{figure} img/pairwise-vs-oif-bindings.svg

Schematic comparison of two approaches to the problem of mul-
tiple languages/multiple implementations. **A** Standard pairwise
bindings **B** Bindings via Open Interfaces (OIF).
```

With the help of _Open Interfaces_, this responsibility can be taken from the
user, if this particular solver is supported.

Then the necessary data conversions between Python and C are provided
automatically, and the time to running the solver from Python is much less.

Moreover, when some particular solver is supported in Open Interfaces,
then the bindings are provided in several languages such as Python, Julia,
C, C++.

Because the heart of the Open Interfaces library is in the library that
provides automatic conversion between different languages, it allows
to avoid a situation where for each particular algorithm implementation
the authors need to provide bindings for several scientific computing
languages, which can be very time-consuming.

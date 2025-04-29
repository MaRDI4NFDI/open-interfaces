# Example: Solve van der Pol equation using Python

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

# 2025-11-20 Optimization interface

Optimization interface should accommodate adapting implementations
for general constrained nonlinear programming:
```math
    minimize_{x \in \mathbb R^n} &f(x)
    subject to & l <= x <= u,
               & l2 <= g(x) <= u2,
```
where $f(x) : \mathbb R^n \to \mathbb R$ and $g(x)$ are nonlinear functions of $x$.

## Julia package JuMP.jl

Julia package /JuMP/ (Julia Mathematical Programming) is a metapackage
that contains a lot of packages for different types of optimization problems:
for example, for integer programming, for linear programming, nonlinear, etc.
It also supports solvers with commercial licenses.

See:
https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP

A part of JuMP is `MathOptInterface` that hides the differences
behind a common interface.

Let's try to solve some simple problem using IPOpt which is continuous
nonlinear programming solver.

The problem is
```math
minimize \sum_{i=1}^{N}x_i^2
```
which is convex, so we should be able to converge to the solution
from any initial guess.

The actual program in Julia would be
```julia
using JuMP
using Ipopt

model = Model(Ipopt.Optimizer)
set_attrbitute(model, "output_flag", false)

# Variables
@variable(model, x)

@objective(model, Min, sum(x.^2))

optimize!(model)
```

One can check the final status of the optimization process via
```
is_solved_and_feasible(model)

termination_status(model)
```
Here, `termination_status` returns constants from the `MathOptInterface`:
there are constants like `OPTIMAL` (global solution),
`LOCALLY_SOLVED` (local minimum), `INFEASIBLE`, etc.

Statements for defining optimizing variables are macros:
```
@variable(model, x)
```
so here `x` is a symbol, and the whole thing is transformed to actual code
(probably, something like `x = variable(model, 'x')`; they do not explain it).

To set constraints or vector variables:
```
@variable(model, -5 <= x[i=1:42] <= 7)
```
Also, upper and lower bounds can be set via keywords arguments to this macro.

Interestingly, when I do
```
typeof(x)
```
then `x` is `VariableRef`.


## SciPy Optimize

It has multiple solvers, although they have slightly different interfaces
and features: some are working with constrained optimization, some do not.

Solution happens through a single function:
```python
from scipy import optimize

x = minimize(
    f,                     # Objective function
    x0,                    # Initial guess
    method="method-name",  # Methods are below
    args=(a, b, c, ...),   # Args that are passed unfolded to f
    jac=None,              # Callback | True, if f returns obj and jac together
    hess=None,             # Callback for Hessian matrix
    hessp=None,            # Callback for computing Hessian-vector product
    options={},            # Dictionary of options
)
```

The interface is general, and not all solvers (methods) use all arguments.

Method names are case-insensitive.

Solvers that use only the objective:
- `Nelder-Mead`
- `Powell`

Solvers that use objective and Jacobian:
- `BFGS` Broyden-Fletcher-Goldfarb-Shanno
  (can estimate Jacobian via finite differences)

Take also Hessian or hessian product
- `Newton-CG` (Newton Conjugate Gradient) H or Hp
- `trust-NCG` Trust-region Newton-Conjugate Gradient
- `trust-Krylov`

Only Hessian:
- `trust-exact` Trust-region Nearly Exact Algorithm: decomposes Hessian via
    Cholesky factorization

### Scipy Optimize: Constrained minimization

Methods are:
- `trust-constr`
- `SLSQP`
- `COBYLA`
- `COBYQA`

**Complication**: they use different interfaces to specify constraints:
`SLSQP` uses a dictionary, while others `LinearConstraint` and
`NOnlinearConstraint` instances.

Linear constraints are written as
\[
\begin{matrix}
c_1^\ell \\
\dots
c_L^\ell
\end{matrix}
\leq
A x
\leq
\begin{matrix}
c_1^u \\
\dots
c_L^u
\end{matrix}
\]

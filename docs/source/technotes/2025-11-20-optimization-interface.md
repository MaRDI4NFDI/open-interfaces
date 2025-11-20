# Optimization interface

## Julia package JuMP.jl

Julia package /JuMP/ (Julia Mathematical Programming) is a metapackage
that contains a lot of packages for different types of optimization problems:
for example, for integer programming, for linear programming, nonlinear, etc.
It also supports solvers with commerical licenses.

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
using IpOpt

model = Model(Ipopt.Optimizer)

# Variables
@variable(model, x)

@objective(model, Min, sum(x.^2))

optimize!(model)
```


# 2024-01-25 Comparison of performance between using `scipy.integrate.ode` versus `ivp` interface

Here we provide comparison of performance differences between "native"
approach, when user code uses numerical packages provided by their language
of choice, and open-interfaces approach, when the user code interacts
with numerical packages via `liboif`.

To conduct the comparison, we solve the initial-value problem
for the inviscid Burgers' equation with periodic boundary conditions:
```{math}
:label: problem

\begin{align}
    u_t + \left(\frac{u^2}{2}\right)_x &= 0,
        \quad x \in [0, 2], \quad t \in [0, 2], \\
    u(0, x) &= 0.5 - 0.25 \sin(\pi x), \\
    u(t, 0) &= u(t, 2)
\end{align}
```
using the method-of-lines approach.
In this approach, one converts a partial differential equation to a system
of ordinary differential equations that can be integrated by third-party
solvers for initial-value problems (time integrators).

We use the following three different implementations for time integration:
- via `ivp` interface using `scipy_ode_dopri5` implementation
- via `ivp` interface using `sundials_cvode` implementation
- via direct use of `scipy.integrate.ode` package with `dopri5` method
  (called `native_scipy_dopri5` in the results below)

Note that the first and the last implementations in the list above are the
same (Runge--Kutta method of 5th order with embedded
4th order method for error estimation by Dormand & Prince).
Besides, `dopri5` method from the `scipy.integrate.ode` package is actually
a wrapper over Fortran implementation, so it is not purely native.

To ensure statistically meaningful results, we integrate with each
implementation multiple times and report the average time-to-solution
as well as standard deviation.
Besides that, we analyze the scalability, that is, how performance
changes with the increase of the problem size (in this case, grid resolution
$N$).

As the problem {eq}`problem` is based on a hyperbolic PDE,
increase of the grid resolution $N$ means proportional decrease
of the maximum allowed time step for numerical stability,
which means that with the increase of the number of grid points
the number of invocations of integration functions increases as well.

To produce the results below, we run the following command:
```shell
python examples/compare_performance_ivp_burgers_eq.py all --n_runs 10
```
where `--n_runs` specifies number of runs for obtaining statistics for each
implementation and grid resolution.

Note that all implementations used absolute and relative tolerances
set to $10^{-15}$.

The results:
```
Statistics:
N = 101
scipy_ode_dopri5           0.42   0.01
sundials_cvode             0.46   0.01
native_scipy_dopri5        0.41   0.01
N = 1001
scipy_ode_dopri5           2.93   0.07
sundials_cvode             4.28   0.12
native_scipy_dopri5        2.77   0.03
N = 10001
scipy_ode_dopri5          78.06   2.61
sundials_cvode           122.70   1.34
native_scipy_dopri5       72.16   0.64
```

This is a debug variant, in which Python callable is replaced by a C function
for `scipy_ode_dopri5`:
```
Statistics:
N = 101
scipy_ode_dopri5           1.40   0.10
sundials_cvode             0.46   0.01
native_scipy_dopri5        0.39   0.02
N = 1001
scipy_ode_dopri5           7.39   0.09
sundials_cvode             4.28   0.05
native_scipy_dopri5        2.75   0.07
N = 10001
scipy_ode_dopri5         107.32   1.95
sundials_cvode           126.12   1.65
```


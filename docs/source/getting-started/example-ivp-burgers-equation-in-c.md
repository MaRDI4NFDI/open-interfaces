# Example: Solving inviscid Burgers' equation from C using IVP interface

Here we demonstrate how to use the `ivp` (initial-value problem for ordinary
differential equations, that is, time integration) interface
from C to solve the following problem based on
the inviscid Burgers' equation:
$$
    \begin{align}
        u_t + \left(\frac{u^2}{2}\right)_x &= 0,
            \quad x \in [0, 2], \quad t \in [0, 2], \\
        u(0, x) &= 0.5 - 0.25 \sin(\pi x), \\
        u(t, 0) &= u(t, 2)
    \end{align}
$$

This is one of the most famous nonlinear hyperbolic equations with the property
that some initial conditions, although smooth, can lead to jump discontinuities
in the solution at some final time.
These jump discontinuities are called __shock waves__.
The initial condition that we choose is exactly of this type, so that at
the final time $T = 2$ the solution develops a shock wave.

To deal with such properties, special numerical methods are used that smooth
the discontinuity a bit so that the computation process remains stable.

Also, it is common to use special time-integration schemes with an upper
bound on the time step, also to keep things numerically stable.

Here, instead of using such special methods for time integration, we will
use adaptive integration.
To facilitate this, we will convert the partial differential equation
into a system of ODEs:
$$
\frac{du_i}{dt} = -0.5 \frac{\partial u_i^2}{\partial x}
$$
which is commonly discretized in space (the right-hand side) in the following
way:
```{math}
:label: ode-system

\frac{du_i}{dt} = - \frac{\hat f(u_{i + 1/2}) - \hat f(u_{i - 1/2})}{\Delta x},
```
where $f = 0.5 u^2$ and $\hat f$ is an approximation of $f$.
There are different ways of approximating $f$, we will use one of the simplest
just for the sake of a demonstration.

To simplify the example, we omit error checking here,
however, the complete code with error checking is available
[in the repository][burgers_c_src].
Implementation of the right-hand side is the following C function:
```c
int
rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t;         /* Unused */
    int retval = 1;
    intptr_t N = y->dimensions[0];

    double *u = y->data;
    double *udot = rhs_out->data;

    double dx = *((double *)user_data);

    double *flux = malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        flux[i] = 0.5 * pow(u[i], 2.0);
    }

    double local_sound_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        if (local_sound_speed < fabs(u[i])) {
            local_sound_speed = fabs(u[i]);
        }
    }

    double *flux_hat = malloc((N - 1) * sizeof(double));
    for (int i = 0; i < N - 1; ++i) {
        flux_hat[i] =
            0.5 * (flux[i] + flux[i + 1]) - 0.5 * local_sound_speed * (u[i + 1] - u[i]);
    }

    for (int i = 1; i < N - 1; ++i) {
        udot[i] = -1.0 / dx * (flux_hat[i] - flux_hat[i - 1]);
    }
    double f_rb = 0.5 * (flux[0] + flux[N - 1]) - 0.5 * local_sound_speed * (u[0] - u[N - 1]);
    double f_lb = f_rb;
    udot[0] = -1.0 / dx * (flux_hat[0] - f_lb);
    udot[N - 1] = -1.0 / dx * (f_rb - flux_hat[N - 2]);

    free(flux);
    free(flux_hat);

    return retval;
}
```

Then we define the rest of the problem and compute the initial condition:
```c
    double t0 = 0.0;
    double t_final = 2.0;
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N});
    double dx;
    double dt_max;
    int status = 1;  // Aux variable to check for errors.

    compute_initial_condition_(N, y0, grid, &dx, &dt_max);
```
we we use the function `oif_create_array_f64` to create arrays
of type `OIFArrayF64` that are part of the __Open Interfaces__ library.

Now we can load the implementation of the `ivp` interface, for example,
`scipy_ode` (which will default to the `dopri5` integrator):
```
    const char impl[] = "scipy_ode";
    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
```


Then we pass to the implementation the details of the problem, namely,
the initial condition:
```
    oif_ivp_set_initial_value(implh, y0, t0);
```
the user data, in our case the spatial step `dx` that is needed in
the right-hand side function:
```
    oif_ivp_set_user_data(implh, &dx);
```
the right-hand side function:
```
    oif_ivp_set_rhs_fn(implh, rhs);
```
and the tolerances for the adaptive integrator:
```c
    oif_ivp_set_tolerances(implh, 1e-8, 1e-12);
```

As we solve a hyperbolic problem, we need to limit the time step
to satisfy the CFL condition, hence, we compute the number of steps
we need to take to reach the final time:
```c
    double dt = dt_max;
    int n_time_steps = (int)(t_final / dt + 1);
```

Now we are ready for the actual integration:
```c
for (int i = 0; i < n_time_steps; ++i) {
    double t = t0 + (i + 1) * dt;
    if (t > t_final) {
        t = t_final;
    }
    status = oif_ivp_integrate(implh, t, y);
    if (status) {
        fprintf(stderr, "oif_ivp_integrate returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
}
```

After the integration is done, we can save the results to a file:
```c
FILE *fp = fopen("burgers_solution.txt", "w+e");
for (int i = 0; i < N; ++i) {
    fprintf(fp, "%.8f %.8f\n", grid->data[i], y->data[i]);
}
fclose(fp);
```
where we use the fact that the data of `OIFArrayF64` arrays
(such as `grid` and `y` in this example)
are stored in the `data` field as a pointer to `double`.

Plotting the results is easy, for example, using Python:
```python
import matplotlib.pyplot as plt
import numpy as np

u = np.loadtxt("solution.txt")

plt.figure()
plt.plot(u, "-")
plt.show()
```

The actual solution at the final time $T = 2$ and the initial condition are
shown on the figure.

```{figure} img/ivp_burgers_soln_scipy_ode_dopri5.pdf
:alt: Numerical solution of the problem for Burgers' equation via `IVP` interface of Open InterFaces using `scipy_ode` implementation.

Numerical solution of the problem for Burgers' equation via `IVP` interface
of Open InterFaces using the `scipy_ode` implementation.
```

The complete code is available
[in the repository][burgers_c_src], is compiled during building the source code
and after building can be run as:
```shell
build/examples/call_ivp_from_c_burgers_eq [implementation]
```
where `[implementation]` is the name of the implementation to use,
one of the following:
- `scipy_ode`,
- `sundials_cvode`,
- `jl_diffeq`.

[burgers_c_src]: https://github.com/MaRDI4NFDI/open-interfaces/blob/main/examples/call_ivp_from_c_burgers_eq.c

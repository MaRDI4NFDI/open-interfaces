#pragma once

#include <oif/api.h>

/**
 * Set right hand side of the system of ordinary differential equations.
 */
int oif_ivp_set_rhs_fn(void rhs(double, OIFArrayF64 *y, OIFArrayF64 *y_dot));

/**
 * Set initial value y(t0) = y0.
 */
int oif_ivp_set_initial_value(OIFArrayF64 *y0, double t0);

/**
 * Integrate to time `t` and write the solution to `y`.
 */
int oif_ivp_integrate(double t, OIFArrayF64 *y);

#pragma once

#include <oif/api.h>

typedef int (*oif_ivp_rhs_fn_t)(double, OIFArrayF64 *y, OIFArrayF64 *y_dot);

/**
 * Set right hand side of the system of ordinary differential equations.
 */
int oif_ivp_set_rhs_fn(ImplHandle implh, oif_ivp_rhs_fn_t rhs);

/**
 * Set initial value y(t0) = y0.
 */
int oif_ivp_set_initial_value(ImplHandle implh, OIFArrayF64 *y0, double t0);

/**
 * Integrate to time `t` and write the solution to `y`.
 */
int oif_ivp_integrate(ImplHandle implh, double t, OIFArrayF64 *y);

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

typedef int (*oif_ivp_rhs_fn_t)(double, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data);

/**
 * Set right hand side of the system of ordinary differential equations.
 */
int
oif_ivp_set_rhs_fn(ImplHandle implh, oif_ivp_rhs_fn_t rhs);

/**
 * Set initial value y(t0) = y0.
 */
int
oif_ivp_set_initial_value(ImplHandle implh, OIFArrayF64 *y0, double t0);

/**
 * Set user data that can be used to pass additional information
 * to the right-hand side function.
 */
int
oif_ivp_set_user_data(ImplHandle implh, void *user_data);

/**
 * Integrate to time `t` and write the solution to `y`.
 */
int
oif_ivp_integrate(ImplHandle implh, double t, OIFArrayF64 *y);

#ifdef __cplusplus
}
#endif


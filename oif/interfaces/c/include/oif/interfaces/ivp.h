/**
 * @file
 * @brief Interface for solving initial-value problems (IVP) for ordinary differential
 * equations (ODE).
 *
 * This interface defines the interface for solving initial-value problems
 * for ordinary differential equations:
 * \f[
 *     \frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0,
 * \f]
 * where \f$y\f$ is the state vector, \f$t\f$ is the time,
 * \f$f\f$ is the right-hand side (RHS) function.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

/**
 * User-provided right-hand side function for the system of ODEs.
 *
 * @param[in] t Current time
 * @param[in] y State vector at time `t`
 * @param[out] ydot Derivative of the state vector, which must be computed during the function
 * call
 * @param[in] user_data User data (additional context) required by the right-hand side function
 */
typedef int (*oif_ivp_rhs_fn_t)(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data);

/**
 * Set right-hand side of the system of ordinary differential equations.
 *
 * @param[in] implh Implementation handle
 * @param[in] rhs Right-hand side callback function
 */
int
oif_ivp_set_rhs_fn(ImplHandle implh, oif_ivp_rhs_fn_t rhs);

/**
 * Set initial value :math:`y(t0) = y0`.
 *
 * @param[in] implh Implementation handle
 * @param[in] y0 Initial value
 * @param[in] t0 Initial time
 */
int
oif_ivp_set_initial_value(ImplHandle implh, OIFArrayF64 *y0, double t0);

/**
 * Set user data (additional context) to pass additional information
 * to the right-hand side function.
 *
 * User data can be any object that is needed by the right-hand side function.
 * For example, if only a scalar value is required as an additional parameter,
 * then `user_data` can be just the pointer to that value.
 * If multiple values are required, then `user_data` can be a pointer to a
 * structure defined by the user, and the it is the user's responsibility
 * to cast the pointer to the correct type in the right-hand side function.
 *
 * @param[in] implh Implementation handle
 * @param[in] user_data User data (pointer to a user-defined object)
 */
int
oif_ivp_set_user_data(ImplHandle implh, void *user_data);

/**
 * Set tolerances for adaptive time integration.
 *
 * @param[in] implh Implementation handle
 * @param[in] rtol Relative tolerance
 * @param[in] atol Absolute tolerance
 */
int
oif_ivp_set_tolerances(ImplHandle implh, double rtol, double atol);

/**
 * Integrate to time `t` and write the solution to `y`.
 *
 * @param[in] implh Implementation handle
 * @param[in] t Time at which we want the solution
 * @param[out] y Array to which the solution at time `t` will be written
 */
int
oif_ivp_integrate(ImplHandle implh, double t, OIFArrayF64 *y);

/**
 * Set integrator and optionally its parameters.
 *
 * Many implementations of ODE solvers contain multiple integrators,
 * with each integrator having specific options.
 * Hence, this function allows the user to set these options for a particular
 * integrator.
 * The integrator name and the options must be checked in the documentation
 * of a particular implementation.
 *
 * @param[in] implh Implementation handle
 * @param[in] integrator_name Name of the integrator
 * @param[in] config Configuration dictionary for the integrator
 */
int
oif_ivp_set_integrator(ImplHandle implh, char *integrator_name, OIFConfigDict *config);

/**
 * Print statistics about integration.
 *
 * @param[in] implh Implementation handle
 */
int
oif_ivp_print_stats(ImplHandle implh);

#ifdef __cplusplus
}
#endif

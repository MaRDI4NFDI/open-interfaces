// clang-format Language: C
#pragma once

#include <oif/api.h>
#include <oif/config_dict.h>

typedef struct self Self;

/**
 * Signature of the right-hand side function f for the system of ODEs.
 */
typedef int (*rhs_fn_t)(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data);

/**
 * Allocate and initialize a new `Self` object.
 * This object is used to store the state of the implementation.
 *
 * @return Pointer to the allocated `Self` object.
 */
Self *
malloc_self(void);

/**
 * Set right hand side of the system of ordinary differential equations.
 */
int
set_rhs_fn(Self *self, rhs_fn_t rhs);

/**
 * Set initial value y(t0) = y0.
 */
int
set_initial_value(Self *self, OIFArrayF64 *y0, double t0);

/**
 * Set user data that can be used to pass additional information
 * to the right-hand side function.
 */
int
set_user_data(Self *self, void *user_data);

/**
 * Integrate to time `t` and write the solution to `y`.
 */
int
integrate(Self *self, double t, OIFArrayF64 *y);

/**
 * Set integrator and optionally its parameters.
 */
int
set_integrator(Self *self, const char *integrator_name, OIFConfigDict *config);

/**
 * Print statistics about integration process.
 */
int
print_stats(Self *self);

/**
 * Free resources allocated for the `Self` object.
 * @param self Pointer to the `Self` object to free.
 */
void
free_self(Self *self);

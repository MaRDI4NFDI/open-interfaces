#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

/**
 * Solve system of linear equations Ax = b via LU factorization.
 */
int
oif_solve_linear_system(ImplHandle implh, OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x);

#ifdef __cplusplus
}
#endif

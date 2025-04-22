/**
 * @file
 * This module defines the interface for solving linear systems of equations.
 *
 * Problems to be solved are of the form:
 *
 * \f[
 *     A x = b,
 * \f]
 * where \f$A\f$ is a square matrix and \f$b\f$ is a vector.
 */
// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

/**
 * Solve the linear system of equations \f$A x = b\f$.

 * @param[in] implh Implementation handle
 * @param[in] A Coefficient matrix with shape `(n, n)`
 * @param[in] b Right-hand side vector with shape `(n,)`
 * @param[out] x Solution vector with shape `(n,)`
 */
int
oif_solve_linear_system(ImplHandle implh, OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x);

#ifdef __cplusplus
}
#endif

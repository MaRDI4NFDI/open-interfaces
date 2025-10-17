// clang-format Language: C
/**
 * @file
 * This module defines the interface for solving a quadratic equation.
 *
 * The quadratic equation is of the form:
 * \f[
 *     a x^2 + b x + c = 0,
 * \f]
 * where \f$a\f$, \f$b\f$, and \f$c\f$ are the coefficients of the equation.
 *
 * Of course, this is not very useful in scientific context to invoke
 * such a solver.
 *
 * It was developed as a prototype to ensure that the envisioned architecture
 * of Open Interfaces is feasible.
 * It is used as a simple text case as well.
 *
 */
// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

/**
 * Solve the quadratic equation \f$a x^2 + b x + c = 0\f$.
 *
 * @param[in] implh Implementation handle
 * @param[in] a Coefficient of the quadratic term
 * @param[in] b Coefficient of the linear term
 * @param[in] c Constant term
 * @param[out] roots Roots of the quadratic equation
 */
int
oif_solve_qeq(ImplHandle implh, double a, double b, double c, OIFArrayF64 *roots);

#ifdef __cplusplus
}
#endif

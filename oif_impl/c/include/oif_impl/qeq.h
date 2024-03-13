#ifndef _OIF_IMPL_QEQ_H_
#define _OIF_IMPL_QEQ_H_

#include <oif/api.h>

/**
 * Solve quadratic equation ax**2 + bx + c = 0
 * Assumes that the output array `roots` always has two elements,
 * so if the roots are repeated, they both will be still present.
 * @param a Coefficient before x**2
 * @param b Coefficient before x
 * @param c Free term
 * @param roots Array with two elements to which the found roots are written.
 * @return int
 */
int
solve_qeq(double a, double b, double c, OIFArrayF64 *roots);
#endif

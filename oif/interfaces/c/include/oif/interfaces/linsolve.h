#pragma once

#include <oif/api.h>


/**
 * Solve system of linear equations Ax = b via LU factorization.
 */
int
oif_solve_linear_system(
    BackendHandle bh, OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x
);

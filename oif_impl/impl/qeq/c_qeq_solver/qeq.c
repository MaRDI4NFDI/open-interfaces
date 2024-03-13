#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <oif_impl/qeq.h>

#include "_qeq.c"

int
solve_qeq(double a, double b, double c, OIFArrayF64 *roots)
{
    int status = solve_qeq_v1(a, b, c, roots->data);
    return status;
}

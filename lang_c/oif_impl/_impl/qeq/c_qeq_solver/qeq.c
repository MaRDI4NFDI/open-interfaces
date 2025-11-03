#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <oif_impl/qeq.h>

#include "_qeq.c"

// We do not actually need to store any data in the `Self` structure,
// hence it is an empty object.
typedef struct self {
} Self;

int
solve_qeq(Self *self, double a, double b, double c, OIFArrayF64 *roots)
{
    (void)self;
    int status = solve_qeq_v1(a, b, c, roots->data);
    return status;
}

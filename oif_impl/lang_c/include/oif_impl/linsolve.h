// clang-format Language: C
#pragma once
#include <oif/api.h>

typedef struct self Self;

int
solve_lin(Self *self, OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x);

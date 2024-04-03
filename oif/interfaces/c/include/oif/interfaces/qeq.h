#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <oif/api.h>

int
oif_solve_qeq(ImplHandle implh, double a, double b, double c, OIFArrayF64 *roots);

#ifdef __cplusplus
}
#endif

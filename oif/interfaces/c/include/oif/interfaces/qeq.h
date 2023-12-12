#pragma once

#include <oif/api.h>

int oif_solve_qeq(
    ImplHandle implh, double a, double b, double c, OIFArrayF64 *roots);

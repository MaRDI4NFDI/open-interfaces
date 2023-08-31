#pragma once

#include <oif/api.h>

int oif_solve_qeq(
    BackendHandle bh, double a, double b, double c, OIFArray *roots
);

#pragma once
#include <oif/api.h>

ImplHandle
oif_init_backend(
    const char *interface, const char *impl, int major, int minor
);

OIFArrayF64 *
create_array_f64(int nd, intptr_t *dimensions);

OIFArrayF64 *
init_array_f64_from_data(int nd, intptr_t *dimensions, double *data);

void
free_array_f64(OIFArrayF64 *x);

void
oif_print_matrix(OIFArrayF64 *mat);

void
oif_print_vector(OIFArrayF64 *vec);

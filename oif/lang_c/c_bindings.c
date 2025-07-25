#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/c_bindings.h>

#include <oif/util.h>

#include <oif/internal/dispatch.h>

ImplHandle
oif_load_impl(const char *interface, const char *impl, int version_major, int version_minor)
{
    return load_interface_impl(interface, impl, version_major, version_minor);
}

int
oif_unload_impl(ImplHandle implh)
{
    return unload_interface_impl(implh);
}

OIFArrayF64 *
oif_create_array_f64(int nd, const intptr_t *const dimensions)
{
    OIFArrayF64 *x = oif_util_malloc(sizeof(OIFArrayF64));
    x->nd = nd;
    x->dimensions = dimensions;
    x->flags = OIF_ARRAY_C_CONTIGUOUS;

    if (nd == 1) {
        x->flags = OIF_ARRAY_C_CONTIGUOUS | OIF_ARRAY_F_CONTIGUOUS;
    }

    int size = 1;
    for (size_t i = 0; i < nd; ++i) {
        size *= dimensions[i];
    }
    x->data = (double *)oif_util_malloc(size * sizeof(double));

    return x;
}

OIFArrayF64 *
oif_init_array_f64_from_data(int nd, const intptr_t *const dimensions, const double *const data)
{
    OIFArrayF64 *x = oif_create_array_f64(nd, dimensions);
    int size = 1;
    for (size_t i = 0; i < nd; ++i) {
        size *= dimensions[i];
    }
    memcpy(x->data, data, size * sizeof(double));

    return x;
}

void
oif_free_array_f64(OIFArrayF64 *x)
{
    if (x == NULL) {
        fprintf(stderr, "[oif_free_array_f64] Attempt to free NULL pointer\n");
        return;
    }
    if (x->data == NULL) {
        fprintf(stderr, "[oif_free_array_f64] Attempt to free NULL pointer\n");
        return;
    }

    oif_util_free(x->data);
    oif_util_free(x);
    x = NULL;
}

void
oif_print_matrix(const OIFArrayF64 *const mat)
{
    assert(mat->nd == 2);

    size_t m = mat->dimensions[0];
    size_t n = mat->dimensions[1];
    double *data = mat->data;
    printf("[ \n");
    for (size_t i = 0; i < m; ++i) {
        printf("[ ");
        for (size_t j = 0; j < n; ++j) {
            printf("%g, ", data[i * n + j]);
        }
        printf("\b\b ],\n");
    }
    printf("\b\b\b]\n");
}

void
oif_print_vector(const OIFArrayF64 *const vec)
{
    assert(vec->nd == 1);

    printf("[ ");
    for (size_t i = 0; i < vec->dimensions[0]; ++i) {
        printf("%g, ", vec->data[i]);
    }
    printf("\b\b ]\n");
}

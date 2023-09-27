#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <oif/api.h>
#include <oif/dispatch.h>


BackendHandle
oif_init_backend(
    const char *backend, const char *interface, int major, int minor) {
    return load_backend_by_name(backend, interface, major, minor);
}

/* int */
/* oif_call_method( */
/*     BackendHandle bh, const char *method, OIFArgs *in_args, OIFArgs *out_args */
/* ) { */


/*     int status = call_interface_method(bh, method, in_args, out_args); */

/*     if (status) { */
/*         fprintf(stderr, "Could not execute interface method"); */
/*     } */

/*     return 0; */
/* } */


OIFArrayF64 *create_array_f64(int nd, intptr_t *dimensions) {
    OIFArrayF64 *x = malloc(sizeof(OIFArrayF64));
    x->nd = nd;
    x->dimensions = dimensions;

    int size = 1;
    for (size_t i = 0; i < nd; ++i) {
        size *= dimensions[i];
    }
    x->data = (double *) malloc(size * sizeof(double));

    return x;
}

OIFArrayF64 *
init_array_f64_from_data(int nd, intptr_t *dimensions, double *data) {
    OIFArrayF64 *x = create_array_f64(nd, dimensions);
    int size = 1;
    for (size_t i = 0; i < nd; ++i) {
        size *= dimensions[i];
    }
    memcpy(x->data, data, size * sizeof(double));

    return x;
}


void free_array_f64(OIFArrayF64 *x) {
    if (x == NULL) {
        return;
    }
    if (x->data == NULL) {
        return;
    }

    free(x->data);
    free(x);
}

void
oif_print_matrix(OIFArrayF64 *mat) {
    assert(mat->nd == 2);

    long m = mat->dimensions[0];
    long n = mat->dimensions[1];
    double *data = mat->data;
    printf("[ \n");
    for (size_t i = 0; i < mat->dimensions[0]; ++i) {
        printf("[ ");
        for (size_t j = 0; j < mat->dimensions[1]; ++j) {
            printf("%g, ", data[i*n + j]);
        }
        printf("\b\b ],\n");
    }
    printf("\b\b\b]\n");
}

void
oif_print_vector(OIFArrayF64 *vec) {
    assert(vec->nd == 1);

    printf("[ ");
    for (size_t i = 0; i < vec->dimensions[0]; ++i) {
        printf("%g, ", vec->data[i]);
    }
    printf("\b\b ]\n");
}

#include <oif/dispatch.h>


BackendHandle
oif_init_backend(
    const char *backend, const char *interface, int major, int minor) {
    return load_backend_by_name(backend, interface, major, minor);
}


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

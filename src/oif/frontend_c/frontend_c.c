#include <oif/api.h>
#include "dispatch.h"


BackendHandle
init_backend(
    const char *backend, const char *interface, int major, int minor) {
    return load_backend_by_name(backend, interface, major, minor);
}


OIFArray *create_array_f64(int nd, intptr_t *dimensions) {
    OIFArray *x = malloc(sizeof(OIFArray));
    x->nd = nd;
    x->dimensions = dimensions;

    int size = 1;
    for (size_t i = 0; i < nd; ++i) {
        size *= dimensions[i];
    }
    x->data = (char *) malloc(size * sizeof(double));

    return x;
}


void free_array(OIFArray *x) {
    if (x == NULL) {
        return;
    }
    if (x->data == NULL) {
        return;
    }

    free(x->data);
    free(x);
}

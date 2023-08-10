#pragma once
#include <stdlib.h>



// Identifier of the used backend.
typedef size_t BackendHandle;


typedef enum {
    OIF_INT = 1,
    OIF_FLOAT32 = 2,
    OIF_FLOAT64 = 3,
    OIF_STR = 4,
} OIFArgType;


typedef struct {
    size_t num_args;
    OIFArgType *arg_types;
    void **args;
} OIFArgs;


/**
 * Load backend by its name and version information.
 */
BackendHandle load_backend(
        const char *backend_name,
        const char *operation,
        size_t version_major,
        size_t version_minor
);


BackendHandle load_backend_c(
        const char *operation,
        size_t version_major,
        size_t version_minor
);


BackendHandle load_backend_python(
        const char *operation,
        size_t version_major,
        size_t version_minor
);


int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals
);


int run_interface_method_c(const char *method, OIFArgs *args, OIFArgs *retvals);


int run_interface_method_python(const char *method, OIFArgs *args, OIFArgs *retvals);

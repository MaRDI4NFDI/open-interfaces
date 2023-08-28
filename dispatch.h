#pragma once
#include <stdint.h>
#include <stdlib.h>


// Identifier of the used backend.
typedef size_t BackendHandle;


typedef enum {
    OIF_INT = 1,
    OIF_FLOAT32 = 2,
    OIF_FLOAT64 = 3,
    OIF_FLOAT32_P = 4,
    OIF_FLOAT64_P = 5,
    OIF_STR = 6,
} OIFArgType;


typedef struct {
    size_t num_args;
    OIFArgType *arg_types;
    void **arg_values;
} OIFArgs;

// This structure closely follows PyArray_Object that describes NumPy arrays.
typedef struct {
    // Number of dimensions in the array.
    int nd;
    // Size of each axis, i = 0, .., nd-1.
    intptr_t *dimensions;
    // Pointer to actual data.
    char *data;
} OIFArray;

enum
{
    BACKEND_C = 0,
    BACKEND_CXX = 1,
    BACKEND_PYTHON = 2,
    BACKEND_JULIA = 3,
    BACKEND_R = 4,
};

char OIF_BACKEND_C_SO[] = "./liboif_backend_c.so";


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


int run_interface_method_c(const char *method, OIFArgs *in_args, OIFArgs *out_args);

int run_interface_method_python(const char *method, OIFArgs *in_args, OIFArgs *out_args);

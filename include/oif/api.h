#pragma once
#include <stddef.h>
#include <stdint.h>

#include <oif/globals.h>

// Identifier of the used backend.
typedef size_t BackendHandle;

typedef enum {
    OIF_INT = 1,
    OIF_FLOAT32 = 2,
    OIF_FLOAT64 = 3,
    OIF_FLOAT32_P = 4,
    OIF_ARRAY_F64 = 5,
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
    double *data;
} OIFArrayF64;

OIFArrayF64 *create_array_f64(int nd, intptr_t *dimensions);
void free_array_f64(OIFArrayF64 *x);

BackendHandle
oif_init_backend(
    const char *backend, const char *interface, int major, int minor
);

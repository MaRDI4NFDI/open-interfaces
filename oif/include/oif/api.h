#pragma once
#include <stddef.h>
#include <stdint.h>

// Identifier of the used backend.
typedef size_t BackendHandle;

enum
{
    BACKEND_C = 0,
    BACKEND_CXX = 1,
    BACKEND_PYTHON = 2,
    BACKEND_JULIA = 3,
    BACKEND_R = 4,
};
#define OIF_BACKEND_COUNT 5

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

enum {
    OIF_ERROR = 101,
    OIF_BACKEND_INIT_ERROR = 102,
};

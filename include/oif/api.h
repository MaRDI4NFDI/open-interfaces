// clang-format Language: C
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Handle to an instantiated implementation.
typedef int ImplHandle;

/**
 * Starting number for implementation handles.
 * If an implementation was successfully loaded,
 * its handle (ID) is guaranteed to be greater or equal than this number.
 */
enum {
    OIF_IMPL_STARTING_NUMBER = 1000,
};

typedef enum {
    OIF_TYPE_BOOL = 1,

    OIF_TYPE_CHAR = 2,
    OIF_TYPE_I8 = 2,

    OIF_TYPE_UCHAR = 3,
    OIF_TYPE_U8 = 3,

    OIF_TYPE_I32 = 4,
    OIF_TYPE_INT = 4,    // Use int as an alias to int32

    OIF_TYPE_U32 = 5,
    OIF_TYPE_UINT = 5,   // Use uint as an alias to uint32

    OIF_TYPE_I64 = 6,
    OIF_TYPE_U64 = 7,

    OIF_TYPE_F32 = 8,

    OIF_TYPE_F64 = 9,

    OIF_TYPE_ARRAY_F32 = 10,

    OIF_ARRAY_F64 = 11,

    OIF_TYPE_ARRAY_F64 = 11,

    OIF_STR = 12,

    OIF_TYPE_STRING = 12,

    OIF_CALLBACK = 13,
    OIF_TYPE_CALLBACK = 13,

    OIF_USER_DATA = 14,
    OIF_TYPE_POINTER = 14,

    OIF_CONFIG_DICT = 15,

    OIF_TYPE_CONFIG_DICT = 15,
} OIFArgType;

enum {
    OIF_LANG_C = 1,
    OIF_LANG_CXX = 2,
    OIF_LANG_PYTHON = 3,
    OIF_LANG_JULIA = 4,
    OIF_LANG_R = 5,
    OIF_LANG_COUNT = 6,
};

enum {
    OIF_ARRAY_C_CONTIGUOUS = 0x0001,
    OIF_ARRAY_F_CONTIGUOUS = 0x0002,
};

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
    // Flags that describe the array: things like row-major order, etc.
    int flags;
} OIFArrayF64;

// Check if the array is C-contiguous.
#define OIF_ARRAY_C_CONTIGUOUS(arr) ((arr)->flags & OIF_ARRAY_C_CONTIGUOUS)

// Check if the array is Fortran-contiguous.
#define OIF_ARRAY_F_CONTIGUOUS(arr) ((arr)->flags & OIF_ARRAY_F_CONTIGUOUS)

// This structure is used for callback functions.
typedef struct {
    int src;        // Language of the function (one of OIF_LANG_* constants)
    void *fn_p_c;   // Function pointer in C
    void *fn_p_jl;  // Function pointer in Julia
    void *fn_p_py;  // Function pointer in Python
} OIFCallback;

/**
 * Structure that holds a pointer to user data in one of its fields
 * depending on the language.
 * All the fields can be NULL except the field `src` that must be set
 * to the constant of the language of the data origin.
 */
typedef struct {
    int src;   // Language of the data origin (one of `OIF_LANG_*` constants)
    void *c;   // Pointer to data in C
    void *jl;  // Pointer to data in Julia
    void *py;  // Pointer to `PyObject` that holds the data in Python
} OIFUserData;

enum {
    OIF_ERROR = 1,
    OIF_IMPL_INIT_ERROR = 2,
    OIF_BRIDGE_NOT_AVAILABLE_ERROR = 3,
    OIF_IMPL_NOT_AVAILABLE_ERROR = 4,
};

#ifdef __cplusplus
}
#endif

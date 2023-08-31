#pragma once
#include <stdint.h>
#include <stdlib.h>

#include <oif/api.h>

enum
{
    BACKEND_C = 0,
    BACKEND_CXX = 1,
    BACKEND_PYTHON = 2,
    BACKEND_JULIA = 3,
    BACKEND_R = 4,
};
#define OIF_BACKEND_COUNT 5

/// Array containing handles to the opened dynamic libraries for the backends.
/**
 * Array containing handles  to the backend dynamic libraries.
 * If some backend is not open, corresponding element is NULL.
 *
 * Each backend is responsible for populating this array with the library
 * handle.
 */
void *OIF_BACKEND_HANDLES[OIF_BACKEND_COUNT];


/**
 * Load backend by its name and version information.
 */
BackendHandle load_backend_by_name(
        const char *backend_name,
        const char *operation,
        size_t version_major,
        size_t version_minor
);


BackendHandle load_backend(
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


int run_interface_method(const char *method, OIFArgs *in_args, OIFArgs *out_args);

// Dispatch library that is called from other languages, and dispatches it
// to the appropriate backend.
#include "dispatch.h"

#include <stdio.h>
#include <string.h>

enum {
    BACKEND_C,
    BACKEND_CXX,
    BACKEND_PYTHON,
    BACKEND_JULIA,
    BACKEND_R,
};


BackendHandle load_backend(
        const char *backend_name,
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    BackendHandle bh;
    if (strcmp(backend_name, "c") == 0) {
        bh = load_backend_c(operation, version_major, version_minor);
    } else if (strcmp(backend_name, "python") == 0) {
        bh = load_backend_python(operation, version_major, version_minor);
    } else {
        perror("Unknown backend");
        bh = 0;
    }

    return bh;
}


BackendHandle load_backend_c(
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    return BACKEND_C;
}


BackendHandle load_backend_python(
        const char *operation,
        size_t version_major,
        size_t version_minor
) {
    perror("This is not yet implemented correctly");
    exit(EXIT_FAILURE);
    return BACKEND_PYTHON;
}


int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals
) {
    int status;
    switch (bh) {
    case BACKEND_C:
        status = run_interface_method_c(method, args, retvals);
        break;
    case BACKEND_PYTHON:
        status = run_interface_method_python(method, args, retvals);
        break;
    default:
        perror("Unknown backend");
        exit(EXIT_FAILURE);
    }
    return status;
}


int
run_interface_method_c(const char *method, OIFArgs *args, OIFArgs *retvals) {
       //void *handle = dlopen("oif_c") 
       return 0;
}

int run_interface_method_python(const char *method, OIFArgs *args, OIFArgs *retvals) {
    return 1;
}

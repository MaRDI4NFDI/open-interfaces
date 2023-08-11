// Dispatch library that is called from other languages, and dispatches it
// to the appropriate backend.
#include "dispatch.h"

#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <string.h>

enum {
    BACKEND_C,
    BACKEND_CXX,
    BACKEND_PYTHON,
    BACKEND_JULIA,
    BACKEND_R,
};

char OIF_BACKEND_C_SO[] = "./liboif_backend_c.so";


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
        fprintf(stderr, "[dispatch] Cannot load backend: %s\n", backend_name);
        exit(EXIT_FAILURE);
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
    // Start Python interpreter here.
    fprintf(stderr, "This is not yet implemented correctly");
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
        fprintf(stderr, "[dispatch] Cannot call interface on backend handle: '%zu'", bh);
        exit(EXIT_FAILURE);
    }
    return status;
}


int
run_interface_method_c(const char *method, OIFArgs *args, OIFArgs *out_args) {
    void *lib_handle = dlopen(OIF_BACKEND_C_SO, RTLD_LOCAL | RTLD_LAZY); 
    if (lib_handle == NULL) {
        fprintf(stderr, "Cannot load shared library %s\n", OIF_BACKEND_C_SO);
        exit(EXIT_FAILURE);
    }
    void *func = dlsym(lib_handle, method);
    if (func == NULL) {
        fprintf(stderr, "Cannot load interface '%s'\n", method);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "I am definitely here\n");

    int num_args = args->num_args;
    fprintf(stderr, "num_args = %d\n", num_args);
    fflush(stderr);

    ffi_cif cif;
    ffi_type **arg_types = malloc(sizeof(ffi_type) * num_args);
    fprintf(stderr, "Ready to malloc\n");
    void **arg_values = malloc(num_args * sizeof(void *));
    fprintf(stderr, "Malloced\n");

    for (size_t i = 0; i < num_args; ++i) {
        fprintf(stderr, "In a loop\n");
        if (args->arg_types[i] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
        } else {
            fflush(stdout);
            fprintf(stderr, "Unknown arg type: %d", args->arg_types[i]);
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
    }

    if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_args, &ffi_type_uint, arg_types) != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[dispatch] ffi_prep_cif was not OK");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_args; ++i) {
        arg_values[i] = args->args[i];
    }

    ffi_arg result;

    ffi_call(&cif, FFI_FN(func), &result, arg_values);

    printf("Result is %ld\n", result);

    return 0;
}

int run_interface_method_python(const char *method, OIFArgs *args, OIFArgs *retvals) {
    return 1;
}

// Dispatch library that is called from other languages, and dispatches it
// to the appropriate backend.
#include "dispatch.h"

#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


char OIF_BACKEND_C_SO[] =      "./liboif_backend_c.so";
char OIF_BACKEND_PYTHON_SO[] = "./liboif_backend_python.so";


BackendHandle load_backend_by_name(
    const char *backend_name,
    const char *operation,
    size_t version_major,
    size_t version_minor)
{
    BackendHandle bh;
    const char *backend_so;

    if (strcmp(backend_name, "c") == 0)
    {
        bh = BACKEND_C;
        backend_so = OIF_BACKEND_C_SO;
    }
    else if (strcmp(backend_name, "python") == 0)
    {
        bh = BACKEND_PYTHON;
        backend_so = OIF_BACKEND_PYTHON_SO;
    }
    else
    {
        fprintf(stderr, "[dispatch] Cannot load backend: %s\n", backend_name);
        exit(EXIT_FAILURE);
    }

    void *lib_handle = dlopen(backend_so, RTLD_LOCAL | RTLD_LAZY);
    if (lib_handle == NULL)
    {
        fprintf(stderr, "[dispatch] Cannot load shared library '%s'\n", backend_so);
        fprintf(stderr, "Error message: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    if (OIF_BACKEND_HANDLES[bh] != NULL) {
        fprintf(stderr, "[dispatch] Backend handle was already set\n");
        exit(EXIT_FAILURE);
    }

    OIF_BACKEND_HANDLES[bh] = lib_handle;

    BackendHandle (*load_backend_fn)(const char *, size_t, size_t);
    load_backend_fn = dlsym(lib_handle, "load_backend");

    if (load_backend_fn == NULL) {
        fprintf(stderr, "[dispatch] Could not load function %s: %s\n",
            "load_backend", dlerror());
    }

    bh = load_backend_fn(operation, version_major, version_minor);
    return bh;
}

int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals)
{
    int status;
    
    if (OIF_BACKEND_HANDLES[bh] == NULL) {
        fprintf(stderr, "[dispatch] Cannot call interface on backend handle: '%zu'", bh);
        exit(EXIT_FAILURE);
    }

    void *lib_handle = OIF_BACKEND_HANDLES[bh];

    int (*run_interface_method_fn)(const char *, OIFArgs *, OIFArgs *);
    run_interface_method_fn = dlsym(lib_handle, "run_interface_method");
    status = run_interface_method_fn(method, args, retvals);

    if (status) {
        fprintf(
            stderr,
            "[dispatch] ERROR: during execution of open interface "
            "an error occurred\n"
        );
    }
    return status;
}

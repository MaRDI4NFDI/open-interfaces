// Dispatch library that is called from other languages, and dispatches it
// to the appropriate language-specific dispatch.
#include <assert.h>
#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/dispatch.h>
#include <oif/dispatch_api.h>


char OIF_DISPATCH_C_SO[] =      "./liboif_dispatch_c.so";
char OIF_DISPATCH_PYTHON_SO[] = "./liboif_dispatch_python.so";

/**
 * Array of handles to the dynamically loaded libraries
 * for the language-specific dispatches.
 */
void *OIF_DISPATCH_HANDLES[OIF_LANG_COUNT];

// Identifier for the language-specific dispatch library (C, Python, etc.).
typedef unsigned int DispatchHandle;


ImplHandle load_interface_impl(
    const char *interface,
    const char *impl,
    size_t version_major,
    size_t version_minor)
{
    DispatchHandle dh;
    const char *dispatch_lang_so;

    char conf_filename[1024] = "oif_impl/impl/";
    strcat(conf_filename, interface);
    strcat(conf_filename, "/");
    strcat(conf_filename, impl);
    strcat(conf_filename, "/");
    strcat(conf_filename, impl);
    strcat(conf_filename, ".conf");

    FILE *conf_file = fopen(conf_filename, "r");
    if (conf_file == NULL) {
        fprintf(
            stderr,
            "[dispatch] Cannot load conf file '%s'\n", conf_filename
        );
        return -1;
    } else {
        fprintf(
            stderr,
            "[dispatch] Configuration file: %s\n", conf_filename
        );
    }

    ssize_t nread;
    // Temporary buffer to read lines from file.
    const size_t buffer_size = 512;
    char *buffer = malloc(buffer_size * sizeof(char));
    char backend_name[16];
    size_t len = buffer_size;
    nread = getline(&buffer, &len, conf_file);
    if (nread != -1) {
        if (nread > sizeof(backend_name) - 1) {
            fprintf(
                stderr, "Backend name is longer than allocated array\n"
            );
            exit(EXIT_FAILURE);
        }
        // Trim new line character.
        if (buffer[nread - 1] == '\n') {
            buffer[nread - 1] = 0;
        }
        strcpy(backend_name, buffer);
    } else {
        fprintf(
            stderr,
            "[dispatch] Could not read backend line from configuration file\n"
        );
        return -1;
    }
    fprintf(stderr, "[dispatch] Backend name: %s\n", backend_name);

    char impl_details[512];
    nread = getline(&buffer, &len, conf_file);
    if (nread != -1) {
        if (nread > sizeof(impl_details) - 1) {
            fprintf(
                stderr, "Backend name is longer than allocated array\n"
            );
            exit(EXIT_FAILURE);
        }
        // Trim new line character.
        if (buffer[nread - 1] == '\n') {
            buffer[nread - 1] = 0;
        }
        strcpy(impl_details, buffer);
    } else {
        fprintf(
            stderr,
            "[dispatch] Could not read implementation details line "
            "from the configuration file\n"
        );
        return -1;
    }
    fprintf(stderr, "[dispatch] Implementation details: '%s'\n", impl_details);
    free(buffer);

    if (strcmp(backend_name, "c") == 0)
    {
        dh = OIF_LANG_C;
        dispatch_lang_so = OIF_DISPATCH_C_SO;
    }
    else if (strcmp(backend_name, "python") == 0)
    {
        dh = OIF_LANG_PYTHON;
        dispatch_lang_so = OIF_DISPATCH_PYTHON_SO;
    }
    else
    {
        fprintf(stderr, "[dispatch] Implementation has unknown backend: '%s'\n", backend_name);
        exit(EXIT_FAILURE);
    }

    void *lib_handle;
    if (OIF_DISPATCH_HANDLES[dh] == NULL) {
        lib_handle = dlopen(dispatch_lang_so, RTLD_LOCAL | RTLD_LAZY);
        if (lib_handle == NULL)
        {
            fprintf(stderr, "[dispatch] Cannot load shared library '%s'\n", dispatch_lang_so);
            fprintf(stderr, "Error message: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
        OIF_DISPATCH_HANDLES[dh] = lib_handle;
    }
    else {
        lib_handle = OIF_DISPATCH_HANDLES[dh];
    }

    ImplHandle (*load_backend_fn)(const char *, size_t, size_t);
    load_backend_fn = dlsym(lib_handle, "load_backend");

    if (load_backend_fn == NULL) {
        fprintf(stderr, "[dispatch] Could not load function %s: %s\n",
            "load_backend", dlerror());
    }

    ImplHandle implh = load_backend_fn(impl_details, version_major, version_minor);
    assert(implh / 1000 == dh);
    return implh;
}

int call_interface_method(
    ImplHandle implh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals)
{
    int status;
    
    DispatchHandle dh = implh / 1000;
    if (OIF_DISPATCH_HANDLES[dh] == NULL) {
        fprintf(stderr, "[dispatch] Cannot call interface implementation for language id: '%u'", dh);
        exit(EXIT_FAILURE);
    }
    void *lib_handle = OIF_DISPATCH_HANDLES[dh];

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

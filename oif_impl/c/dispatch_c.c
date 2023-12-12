#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>
#include <oif/dispatch.h>

typedef struct {
    ImplInfo base;
    void *impl_lib;
} CImplInfo;

static int IMPL_COUNTER = 0;


ImplInfo *load_backend(
    const char *impl_details,
    size_t version_major,
    size_t version_minor)
{
    // For C implementations, `impl_details` must contain the name
    // of the shared library with the methods implemented as functions.
    void *impl_lib = dlopen(impl_details, RTLD_LOCAL | RTLD_LAZY);
    if (impl_lib == NULL) {
        fprintf(
            stderr,
            "[dispatch_c] Could not load implementation library '%s', error: %s\n",
            impl_details,
            dlerror()
        );
        return NULL;
    }

    CImplInfo *impl_info = malloc(sizeof(CImplInfo));
    if (impl_info == NULL) {
        fprintf(stderr, "[dispatch_c] Could not create an implementation structure\n");
        return NULL;
    }
    impl_info->impl_lib = impl_lib;

    return (ImplInfo *) impl_info;
}


int run_interface_method(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    if (impl_info->dh != OIF_LANG_C) {
        fprintf(stderr, "[dispatch_c] Provided implementation is not implemented in C\n");
        return -1;
    }
    CImplInfo *impl = (CImplInfo *) impl_info;
    void *service_lib = impl->impl_lib;
    void *func = dlsym(service_lib, method);
    if (func == NULL)
    {
        fprintf(stderr, "[dispatch_c] Cannot load interface '%s'\n", method);
        fprintf(stderr, "[dispatch_c] dlerror() = %s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    int num_in_args = in_args->num_args;
    int num_out_args = out_args->num_args;
    int num_total_args = num_in_args + num_out_args;

    ffi_cif cif;
    ffi_type **arg_types = malloc(num_total_args * sizeof(ffi_type));
    void **arg_values = malloc(num_total_args * sizeof(void *));

    // Merge input and output argument types together in `arg_types` array.
    for (size_t i = 0; i < num_in_args; ++i)
    {
        if (in_args->arg_types[i] == OIF_FLOAT64)
        {
            arg_types[i] = &ffi_type_double;
        }
        else if (in_args->arg_types[i] == OIF_ARRAY_F64)
        {
            arg_types[i] = &ffi_type_pointer;
        } else if (in_args->arg_types[i] == OIF_CALLBACK) {
            arg_types[i] = &ffi_type_pointer;
            fprintf(stderr, "[dispatch_c] WARN: only C callbacks are processed correctly\n");
            in_args->arg_values[i] = ((OIFCallback *) in_args->arg_values[i])->c_fn_p;
        }
        else
        {
            fflush(stdout);
            fprintf(stderr, "[dispatch_c] Unknown input arg type: %d\n", in_args->arg_types[i]);
            exit(EXIT_FAILURE);
        }
    }
    for (size_t i = num_in_args; i < num_total_args; ++i)
    {
        if (out_args->arg_types[i - num_in_args] == OIF_FLOAT64)
        {
            arg_types[i] = &ffi_type_double;
        }
        else if (out_args->arg_types[i - num_in_args] == OIF_ARRAY_F64)
        {
            arg_types[i] = &ffi_type_pointer;
        }
        else
        {
            fflush(stdout);
            fprintf(stderr, "[dispatch_c] Unknown output arg type: %d\n", out_args->arg_types[i - num_in_args]);
            exit(EXIT_FAILURE);
        }
    }

    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_total_args, &ffi_type_uint, arg_types);
    if (status == FFI_OK)
    {
        // printf("[dispatch_c] ffi_prep_cif returned FFI_OK\n");
    }
    else
    {
        fflush(stdout);
        fprintf(stderr, "[dispatch_c] ffi_prep_cif was not OK");
        exit(EXIT_FAILURE);
    }

    // Merge input and output argument values together in `arg_values` array.
    for (size_t i = 0; i < num_in_args; ++i)
    {
        arg_values[i] = in_args->arg_values[i];
    }
    for (size_t i = num_in_args; i < num_total_args; ++i)
    {
        arg_values[i] = out_args->arg_values[i - num_in_args];
    }

    unsigned result;
    ffi_call(&cif, FFI_FN(func), &result, arg_values);

    // printf("Result is %u\n", result);
    fflush(stdout);

    return 0;
}

#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/dispatch.h>
#include <oif/dispatch_api.h>

/**
 * Duplicate a given null-terminated string along with memory allocation.
 *
 * It is the user responsibility to free the allocated memory.
 * This function is basically a copy of `strdup` because it is not in ISO C.
 *
 * @return The pointer to the duplicated null-terminated string or NULL in case
 * of an error.
 */
static char *
str_duplicate(const char src[static 1])
{
    size_t len = strlen(src);
    char *dest = malloc(len * sizeof(*dest));
    if (dest == NULL) {
        fprintf(stderr, "[str_duplicate] Could not allocate memory\n");
        return NULL;
    }
    strcpy(dest, src);
    return dest;
}

typedef struct {
    ImplInfo base;
    void *impl_lib;
    char *impl_details;
} CImplInfo;

static int IMPL_COUNTER = 0;

ImplInfo *
load_impl(const char *impl_details, size_t version_major, size_t version_minor)
{
    (void) version_major;
    (void) version_minor;
    // For C implementations, `impl_details` must contain the name
    // of the shared library with the methods implemented as functions.
    void *impl_lib = dlopen(impl_details, RTLD_LOCAL | RTLD_LAZY);
    if (impl_lib == NULL) {
        fprintf(stderr,
                "[dispatch_c] Could not load implementation library '%s', "
                "error: %s\n",
                impl_details, dlerror());
        return NULL;
    }

    CImplInfo *impl_info = malloc(sizeof(CImplInfo));
    if (impl_info == NULL) {
        fprintf(stderr, "[dispatch_c] Could not create an implementation structure\n");
        return NULL;
    }
    impl_info->impl_lib = impl_lib;
    impl_info->impl_details = str_duplicate(impl_details);
    assert(impl_info->impl_details != NULL);
    fprintf(stderr, "[dispatch_c] load_impl impl_info->impl_details = %s\n",
            impl_info->impl_details);

    return (ImplInfo *)impl_info;
}

int
unload_impl(ImplInfo *impl_info_)
{
    if (impl_info_->dh != OIF_LANG_C) {
        fprintf(stderr,
                "[dispatch_python] unload_impl received non-C implementation "
                "argument\n");
        return -1;
    }
    CImplInfo *impl_info = (CImplInfo *)impl_info_;

    int status = dlclose(impl_info->impl_lib);
    if (status != 0) {
        fprintf(stderr,
                "[dispatch_c] While closing implementation '%s' an error occurred. "
                "Error message: %s",
                impl_info->impl_details, dlerror());
    }
    IMPL_COUNTER--;

    free(impl_info->impl_details);
    free(impl_info);
    return 0;
}

int
call_impl(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    int result = 1;
    ffi_cif cif;
    ffi_type **arg_types = NULL;
    void **arg_values = NULL;

    if (impl_info->dh != OIF_LANG_C) {
        fprintf(stderr, "[dispatch_c] Provided implementation is not implemented in C\n");
        return -1;
    }

    CImplInfo *impl = (CImplInfo *)impl_info;
    void *service_lib = impl->impl_lib;
    void *func = dlsym(service_lib, method);
    if (func == NULL) {
        fprintf(stderr, "[dispatch_c] Cannot load interface '%s'\n", method);
        fprintf(stderr, "[dispatch_c] dlerror() = %s\n", dlerror());
        goto cleanup;
    }

    size_t num_in_args = in_args->num_args;
    size_t num_out_args = out_args->num_args;
    unsigned int num_total_args = (unsigned int) (num_in_args + num_out_args);

    arg_types = malloc(num_total_args * sizeof(ffi_type *));
    if (arg_types == NULL) {
        fprintf(stderr, "[dispatch_c] Could not allocate memory for FFI types\n");
        goto cleanup;
    }
    arg_values = malloc(num_total_args * sizeof(void *));
    if (arg_values == NULL) {
        fprintf(stderr, "[dispatch_c] Could not allocate memory for FFI values\n");
        goto cleanup;
    }

    // Merge input and output argument types together in `arg_types` array.
    for (size_t i = 0; i < num_in_args; ++i) {
        if (in_args->arg_types[i] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
        }
        else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
            arg_types[i] = &ffi_type_pointer;
        }
        else if (in_args->arg_types[i] == OIF_USER_DATA) {
            arg_types[i] = NULL;
            arg_types[i] = &ffi_type_pointer;
            OIFUserData *user_data = (OIFUserData *) in_args->arg_values[i];
            if (user_data->src == OIF_LANG_C) {
                in_args->arg_values[i] = &user_data->c;
            }
            else if (user_data->src == OIF_LANG_PYTHON) {
                in_args->arg_values[i] = &user_data->py;
            }
            else {
                fprintf(stderr, "[dispatch_c] Cannot handle OIFUserData because of the unsupported language.\n");
                goto cleanup;
            }
        }
        else if (in_args->arg_types[i] == OIF_CALLBACK) {
            arg_types[i] = &ffi_type_pointer;
            // We need to take a pointer to a pointer according to the FFI
            // convention, hence the & operator.
            in_args->arg_values[i] = &((OIFCallback *)in_args->arg_values[i])->fn_p_c;
        }
        else {
            fflush(stdout);
            fprintf(stderr, "[dispatch_c] Unknown input arg type: %d\n",
                    in_args->arg_types[i]);
            goto cleanup;
        }
    }
    for (size_t i = num_in_args; i < num_total_args; ++i) {
        if (out_args->arg_types[i - num_in_args] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
        }
        else if (out_args->arg_types[i - num_in_args] == OIF_ARRAY_F64) {
            arg_types[i] = &ffi_type_pointer;
        }
        else {
            fprintf(stderr, "[dispatch_c] Unknown output arg type: %d\n",
                    out_args->arg_types[i - num_in_args]);
            goto cleanup;
        }
    }

    ffi_status status =
        ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_total_args, &ffi_type_sint, arg_types);
    if (status != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[dispatch_c] ffi_prep_cif was not OK");
        goto cleanup;
    }

    // Merge input and output argument values together in `arg_values` array.
    for (size_t i = 0; i < num_in_args; ++i) {
        arg_values[i] = in_args->arg_values[i];
    }
    for (size_t i = num_in_args; i < num_total_args; ++i) {
        arg_values[i] = out_args->arg_values[i - num_in_args];
    }

    ffi_call(&cif, FFI_FN(func), &result, arg_values);

cleanup:
    if (arg_values != NULL) {
        free(arg_values);
    }
    if (arg_types != NULL) {
        free(arg_types);
    }

    return result;
}

#include <assert.h>
#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include <oif/util.h>
#include <oif/_platform.h>

#include <oif/internal/bridge_api.h>

#include "allocation_tracker.h"

typedef struct {
    ImplInfo base;
    void *self;
    void *impl_lib;
    char *impl_details;
} CImplInfo;

static int IMPL_COUNTER_ = 0;

static char *prefix_ = "bridge_c";

ImplInfo *
load_impl(const char *interface, const char *impl_details, size_t version_major, size_t version_minor)
{
    (void)version_major;
    (void)version_minor;
    // For C implementations, `impl_details` must contain the name
    // of the shared library with the methods implemented as functions.
    void *impl_lib = dlopen(impl_details, RTLD_LOCAL | RTLD_LAZY);
    if (impl_lib == NULL) {
        fprintf(stderr,
                "[%s] Could not load implementation library '%s', "
                "error: %s\n",
                prefix_, impl_details, dlerror());
        return NULL;
    }

    CImplInfo *impl_info = oif_util_malloc(sizeof(CImplInfo));
    if (impl_info == NULL) {
        fprintf(stderr, "[%s] Could not create an implementation structure\n", prefix_);
        dlclose(impl_lib);
        return NULL;
    }

    size_t create_fn_name_len =
        strlen("oif_") + strlen(interface) + strlen("_create") + 1;
    char *create_fn_name = oif_util_malloc(create_fn_name_len);
    sprintf(create_fn_name, "%s%s%s", "oif_", interface, "_create");
    void *(*create_fn)() = dlsym(impl_lib, create_fn_name);

    if (create_fn != NULL) {
        impl_info->self = create_fn();
        if (impl_info->self == NULL) {
            fprintf(stderr,
                    "[%s] Implementation '%s' returned NULL from method '%s'\n",
                    prefix_, impl_details, create_fn_name);
            free(create_fn_name);
            dlclose(impl_lib);
            oif_util_free(impl_info);
            return NULL;
        }
    }
    else {
        fprintf(stderr,
                "[%s] Implementation '%s' does not implement method '%s'\n",
                prefix_, impl_details, create_fn_name);
        impl_info->self = NULL;
    }
    oif_util_free(create_fn_name);

    impl_info->impl_lib = impl_lib;
    impl_info->impl_details = oif_util_str_duplicate(impl_details);
    assert(impl_info->impl_details != NULL);
    fprintf(stderr, "[dispatch_c] load_impl impl_info->impl_details = %s\n",
            impl_info->impl_details);

    IMPL_COUNTER_++;
    return (ImplInfo *)impl_info;
}

int
unload_impl(ImplInfo *impl_info_)
{
    int status;
    if (impl_info_->dh != OIF_LANG_C) {
        fprintf(stderr,
                "[%s] unload_impl received non-C implementation "
                "argument\n",
                prefix_);
        return -1;
    }
    CImplInfo *impl_info = (CImplInfo *)impl_info_;

    size_t free_fn_name_len =
        strlen("oif_") + strlen(impl_info_->interface) + strlen("_free") + 1;
    char *free_fn_name = malloc(free_fn_name_len);
    sprintf(free_fn_name, "%s%s%s", "oif_", impl_info_->interface, "_free");
    void *free_fn = dlsym(impl_info->impl_lib, free_fn_name);

    if (free_fn == NULL) {
        logwarn(prefix_, "Implementation '%s' does not implement method '%s'\n",
                impl_info->impl_details, free_fn_name);
    }
    else {
        OIFArgType in_arg_types[] = {OIF_USER_DATA};
        void **in_arg_values = {NULL};
        OIFArgs in_args = {
            .num_args = 0,
            .arg_types = in_arg_types,
            .arg_values = in_arg_values,
        };

        OIFArgs out_args = {
            .num_args = 0,
            .arg_types = NULL,
            .arg_values = NULL,
        };

        fprintf(stderr, "[%s] Calling method '%s' for implementation '%s'\n", prefix_,
                free_fn_name, impl_info->impl_details);
        status = call_impl(impl_info_, "oif_ivp_free", &in_args, &out_args);
        if (status != 0) {
            fprintf(stderr,
                    "[%s] !!! Error occurred while calling method '%s' for "
                    "implementation '%s'\n",
                    prefix_, free_fn_name, impl_info->impl_details);
        }
    }
    free(free_fn_name);

#if !defined(OIF_SANITIZE_ADDRESS_ENABLED)
    status = dlclose(impl_info->impl_lib);
#else
    status = 0;
#endif
    if (status != 0) {
        fprintf(stderr,
                "[%s] While closing implementation '%s' an error occurred. "
                "Error message: %s",
                prefix_, impl_info->impl_details, dlerror());
    }
    IMPL_COUNTER_--;

    oif_util_free(impl_info->impl_details);
    return 0;
}

int
call_impl(ImplInfo *impl_info_, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    int result = 1;
    ffi_cif cif;
    ffi_type **arg_types = NULL;
    void **arg_values = NULL;
    AllocationTracker *tracker = NULL;

    if (impl_info_->dh != OIF_LANG_C) {
        fprintf(stderr, "[%s] Provided implementation is not implemented in C\n", prefix_);
        return -1;
    }

    CImplInfo *impl_info = (CImplInfo *)impl_info_;
    void *service_lib = impl_info->impl_lib;
    void *func = dlsym(service_lib, method);
    if (func == NULL) {
        fprintf(stderr, "[%s] Cannot load method '%s'\n", prefix_, method);
        fprintf(stderr, "[%s] dlerror() = %s\n", prefix_, dlerror());
        goto cleanup;
    }

    size_t num_in_args = in_args->num_args;
    size_t num_out_args = out_args->num_args;
    // We need to add one more argument for the `Self *self` argument.
    unsigned int num_total_args = (unsigned int)(1 + num_in_args + num_out_args);

    arg_types = oif_util_malloc(num_total_args * sizeof(ffi_type *));
    if (arg_types == NULL) {
        fprintf(stderr, "[%s] Could not allocate memory for FFI types\n", prefix_);
        goto cleanup;
    }
    arg_values = oif_util_malloc(num_total_args * sizeof(void *));
    if (arg_values == NULL) {
        fprintf(stderr, "[%s] Could not allocate memory for FFI values\n", prefix_);
        goto cleanup;
    }

    arg_types[0] = &ffi_type_pointer;  // Self *self
    arg_values[0] = &impl_info->self;   // Pass the `Self *self` argument
                                        // which for libffi
                                        // must be passed as `Self **self`.

    tracker = allocation_tracker_init();

    // Merge input and output argument types together in `arg_types` array.
    for (unsigned int i = 0; i < num_in_args; ++i) {
        if (in_args->arg_types[i] == OIF_FLOAT64) {
            arg_types[i + 1] = &ffi_type_double;
        }
        else if (in_args->arg_types[i] == OIF_ARRAY_F64 || in_args->arg_types[i] == OIF_STR) {
            arg_types[i + 1] = &ffi_type_pointer;
        }
        else if (in_args->arg_types[i] == OIF_CALLBACK) {
            arg_types[i + 1] = &ffi_type_pointer;
            // We need to take a pointer to a pointer according to the FFI
            // convention, hence the & operator.
            in_args->arg_values[i] = &((OIFCallback *)in_args->arg_values[i])->fn_p_c;
        }
        else if (in_args->arg_types[i] == OIF_USER_DATA) {
            arg_types[i + 1] = &ffi_type_pointer;
            OIFUserData *user_data = (OIFUserData *)in_args->arg_values[i];
            if (user_data->src == OIF_LANG_C) {
                in_args->arg_values[i] = &user_data->c;
            }
            else if (user_data->src == OIF_LANG_PYTHON) {
                in_args->arg_values[i] = &user_data->py;
            }
            else if (user_data->src == OIF_LANG_JULIA) {
                in_args->arg_values[i] = &user_data->jl;
            }
            else {
                fprintf(stderr,
                        "[%s] Cannot handle OIFUserData because of the unsupported "
                        "language.\n",
                        prefix_);
                goto cleanup;
            }
        }
        else if (in_args->arg_types[i] == OIF_CONFIG_DICT) {
            OIFConfigDict *dict = *(OIFConfigDict **)in_args->arg_values[i];
            arg_types[i + 1] = &ffi_type_pointer;
            if (dict != NULL) {
                // We cannot simply assign `&new_dict` to `in_args->arg_values[i]`
                // as it gets trashed as soon as we leave this block.
                // Yes, C programming is amusing.
                OIFConfigDict *new_dict = oif_config_dict_init();

                oif_config_dict_copy_serialization(new_dict, dict);
                oif_config_dict_deserialize(new_dict);

                // We need to obtain a pointer as it is required by `libffi`.
                OIFConfigDict **new_dict_p = malloc(sizeof(void *));
                if (new_dict_p == NULL) {
                    fprintf(stderr, "[%s] Could not allocate memory\n", prefix_);
                    exit(1);
                }
                *new_dict_p = new_dict;
                in_args->arg_values[i] = new_dict_p;
                // Note that the ownership to OIFConfigDict is passed
                // to the callee as it is highly likely that they need
                // to save the dictionary somewhere and use it in another
                // function as initialization of solvers is often spread out
                // between several different function invocations.
                // Hence, the callee is responsible to free the memory
                // after they do not need it anymore.
                /* allocation_tracker_add(tracker, new_dict, oif_config_dict_free); */
                allocation_tracker_add(tracker, new_dict_p, NULL);
            }
        }
        else {
            fflush(stdout);
            fprintf(stderr, "[%s] Unknown input arg type: %d\n", prefix_,
                    in_args->arg_types[i]);
            goto cleanup;
        }
    }

    for (unsigned int i = num_in_args; i < num_total_args - 1; ++i) {
        if (out_args->arg_types[i - num_in_args] == OIF_FLOAT64) {
            arg_types[i + 1] = &ffi_type_double;
        }
        else if (out_args->arg_types[i - num_in_args] == OIF_ARRAY_F64) {
            arg_types[i + 1] = &ffi_type_pointer;
        }
        else {
            fprintf(stderr, "[%s] Unknown output arg type: %d\n", prefix_,
                    out_args->arg_types[i - num_in_args]);
            goto cleanup;
        }
    }

    ffi_status status =
        ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_total_args, &ffi_type_sint, arg_types);
    if (status != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[%s] ffi_prep_cif was not OK", prefix_);
        goto cleanup;
    }

    // Merge input and output argument values together in `arg_values` array.
    for (size_t i = 0; i < num_in_args; ++i) {
        arg_values[i + 1] = in_args->arg_values[i];
    }
    for (size_t i = num_in_args; i < num_total_args - 1; ++i) {
        arg_values[i + 1] = out_args->arg_values[i - num_in_args];
    }

    ffi_call(&cif, FFI_FN(func), &result, arg_values);

cleanup:
    if (arg_values != NULL) {
        oif_util_free(arg_values);
    }
    if (arg_types != NULL) {
        oif_util_free(arg_types);
    }

    if (tracker != NULL) {
        allocation_tracker_free(tracker);
    }

    return result;
}

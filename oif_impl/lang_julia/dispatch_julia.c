#include <assert.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <julia.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>
#include <oif/_platform.h>

static char *prefix_ = "dispatch_julia";

enum { BUFFER_SIZE_ = 32 };

typedef struct {
    ImplInfo base;
    const char *module_name;
    jl_module_t *module;
} JuliaImplInfo;

static void
handle_exception_(void)
{
    jl_value_t *exc = jl_exception_occurred();
    jl_value_t *sprint_fun = jl_get_function(jl_base_module, "sprint");
    jl_value_t *showerror_fun = jl_get_function(jl_base_module, "showerror");

    const char *exc_msg = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exc));
    printf("[%s] ERROR: %s\n", prefix_, exc_msg);
    jl_exception_clear();
}


static size_t
my_strcpy(char dst[static 1], const char *src, size_t size_dst)
{
    char *dst_right = dst + size_dst;

    size_t chars_written = 0;
    while (dst < dst_right && dst != NULL && *src != '\0') {
        *dst = *src;
        dst++;
        src++;
        chars_written++;
    }

    if (dst != NULL) {
        *dst = '\0';
    }
    return chars_written;
}

/**
 * Build a Julia tuple from an `intptr_t` array.
 *
 * Due to complicated Julia C API, the tuple is created via `jl_eval_string`.
 *
 * @param dimensions Array with elements that are non-negative numbers
 * @param ndims Number of the elements in the array
 * @return Julia tuple on success, `NULL` otherwise
 */
static jl_value_t *
build_julia_tuple_from_size_t_array(intptr_t *dimensions, size_t ndims) {
    char *tuple_string = malloc(sizeof(char) * 1024);
    char *cursor = tuple_string;
    char *right_bound = &tuple_string[1024-1];
    my_strcpy(tuple_string, "(", right_bound - cursor);
    cursor++;
    char buffer[BUFFER_SIZE_];
    size_t chars_written;
    jl_value_t *tuple = NULL;
    uintptr_t diff;

    for (size_t i = 0; i < ndims; ++i) {
        chars_written = snprintf(buffer, BUFFER_SIZE_, "%zu", dimensions[i]);
        if (chars_written >= BUFFER_SIZE_) {
            goto report_too_long;
        }
        diff = right_bound - cursor;
        chars_written = my_strcpy(cursor, buffer, diff);
        if (chars_written >= diff) {
            goto report_too_long;
        }
        cursor += chars_written;
        if (i < ndims - 1 || ndims == 1) {
            chars_written = my_strcpy(cursor, ",", right_bound - cursor);
            diff = right_bound - cursor;
            if (chars_written >= diff) {
                goto report_too_long;
            }
            cursor++;
        }
    }
    diff = right_bound - cursor;
    chars_written = my_strcpy(cursor, ")", diff);
    if (chars_written >= diff) {
        fprintf(stderr, "ERROR: the string to copy is too long\n");
        exit(1);
    }

    tuple = jl_eval_string(tuple_string);
    if (jl_exception_occurred()) {
        const char *p = jl_string_ptr(jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))"));
        fprintf(stderr, "%s\n", p);
    }
    goto cleanup;

report_too_long:
    fprintf(stderr, "[build_julia_tuple_from_size_t_array] The string representation of the julia tuple does not fit in the buffer\n");

cleanup:
    free(tuple_string);

    return tuple;
}

ImplInfo *
load_impl(const char *impl_details, size_t version_major, size_t version_minor)
{
    (void)impl_details;
    (void)version_major;
    (void)version_minor;
    JuliaImplInfo *result = NULL;

    jl_init();
    static_assert(sizeof(int) == 4, "The code is written in assumption that C int is 32-bit");

    char module_filename[512] = "\0";
    char module_name[512] = "\0";
    size_t i;
    for (i = 0; i < strlen(impl_details); ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            module_filename[i] = impl_details[i];
        }
        else {
            module_filename[i] = '\0';
            break;
        }
    }
    size_t offset = i + 1;
    for (; i < strlen(impl_details); ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            module_name[i - offset] = impl_details[i];
        }
        else {
            module_name[i] = '\0';
        }
    }

    fprintf(stderr, "[%s] Provided module filename: '%s'\n", prefix_, module_filename);
    fprintf(stderr, "[%s] Provided module name: '%s'\n", prefix_, module_name);

    char include_statement[1024];
    sprintf(include_statement, "include(\"oif_impl/impl/%s\")", module_filename);
    printf("Executing in julia: %s\n", include_statement);
    char import_statement[1024];
    sprintf(import_statement, "import .%s", module_name);

    jl_value_t *retval;
    retval = jl_eval_string(include_statement);
    if (jl_exception_occurred()) {
        goto catch;
    }
    jl_static_show(jl_stdout_stream(), retval);
    jl_printf(jl_stdout_stream(), "\n");

    retval = jl_eval_string(import_statement);
    if (jl_exception_occurred()) {
        goto catch;
    }
    jl_static_show(jl_stdout_stream(), retval);
    jl_printf(jl_stdout_stream(), "\n");

    jl_module_t *module = (jl_module_t *)jl_eval_string("QeqSolver");

    result = malloc(sizeof *result);
    if (result == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Julia implementation information\n",
                prefix_);
        goto cleanup;
    }
    result->module_name = module_name;
    result->module = module;

    goto cleanup;

catch:
    handle_exception_();

cleanup:

    return (ImplInfo *)result;
}

int
unload_impl(ImplInfo *impl_info_)
{
    assert(impl_info_->dh == OIF_LANG_JULIA);
    JuliaImplInfo *impl_info = (JuliaImplInfo *) impl_info_;
    free(impl_info);

    jl_atexit_hook(0);
    return 0;
}

int
call_impl(ImplInfo *impl_info_, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    int result = -1;

    if (impl_info_->dh != OIF_LANG_JULIA) {
        fprintf(stderr, "[%s] Provided implementation is not in Julia\n", prefix_);
        return -1;
    }
    JuliaImplInfo *impl_info = (JuliaImplInfo *)impl_info_;

    assert(in_args->num_args < INT32_MAX);
    assert(out_args->num_args < INT32_MAX);
    int32_t in_num_args = (int32_t)in_args->num_args;
    int32_t out_num_args = (int32_t)out_args->num_args;
    int32_t num_args = in_num_args + out_num_args;

    jl_value_t **julia_args;
    JL_GC_PUSHARGS(julia_args, num_args);

    double roots[2] = {99.0, 25.0};

    for (int32_t i = 0; i < in_num_args; ++i) {
        if (in_args->arg_types[i] == OIF_FLOAT64) {
            julia_args[i] = jl_box_float64(*(double *) in_args->arg_values[i]);
        }
        else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
            OIFArrayF64 *oif_array = *(OIFArrayF64 **)in_args->arg_values[i];
            jl_value_t *arr_type = jl_apply_array_type((jl_value_t *)jl_float64_type, 1);
            jl_value_t *dims = build_julia_tuple_from_size_t_array(oif_array->dimensions, oif_array->nd);
            bool own_buffer = false;
            julia_args[i] = (jl_value_t *)jl_ptr_to_array(arr_type, roots, (jl_value_t *)dims, own_buffer); 
        } else {
            fprintf(
                stderr,
                "[%s] Cannot convert input argument #%d with "
                "provided type id %d\n",
                prefix_, i, in_args->arg_types[i]
            );
            goto cleanup;
        }
    }

    for (int32_t i = in_num_args; i < num_args; ++i) {
        if (out_args->arg_types[i - in_num_args] == OIF_FLOAT64) {
            julia_args[i] = jl_box_float64(*(float *) in_args->arg_values[i]);
        }
        else if (out_args->arg_types[i - in_num_args] == OIF_ARRAY_F64) {
            OIFArrayF64 *oif_array = *(OIFArrayF64 **)out_args->arg_values[i - in_num_args];
            jl_value_t *arr_type = jl_apply_array_type((jl_value_t *)jl_float64_type, 1);
            jl_value_t *dims = build_julia_tuple_from_size_t_array(oif_array->dimensions, oif_array->nd);
            bool own_buffer = false;
            julia_args[i] = (jl_value_t *)jl_ptr_to_array(arr_type, oif_array->data, (jl_value_t *)dims, own_buffer); 
        }
        else {
            fprintf(
                stderr,
                "[%s] Cannot convert output argument #%d with "
                "provided type id %d\n",
                prefix_, i, in_args->arg_types[i]
            );
            goto cleanup;
        }
    }

    jl_function_t *fn;
    // It is customary for the Julia code to suffix function names with '!'
    // if they modify their arguments.
    // Because the Open Interfaces do not add '!' to the function names,
    // we need to check if the function with the '!' suffix exists.
    if (out_num_args == 0) {
        fn = jl_get_function(impl_info->module, method);
        if (fn == NULL) {
            fprintf(
                stderr,
                "[%s] Could not find method '%s' in implementation with id %d\n",
                prefix_, method, impl_info->base.implh
            );
            goto cleanup;
        }
    }
    else {
        char non_pure_method[64];
        strcpy(non_pure_method, method);
        strcat(non_pure_method, "!");
        fn = jl_get_function(impl_info->module, non_pure_method);
        if (fn == NULL) {
            fprintf(
                stderr,
                "[%s] Could not find method '%s!' in implementation with id %d\n",
                prefix_, method, impl_info->base.implh
            );
            goto cleanup;
        }
    }

    jl_value_t *retval_ = jl_call(fn, julia_args, num_args);
    if (jl_exception_occurred()) {
        handle_exception_();
        goto cleanup;
    }
    int64_t retval = jl_unbox_int64(retval_);
    assert(retval == 0);
    result = 0;

cleanup:
    JL_GC_POP();

finally:
    return result;
}

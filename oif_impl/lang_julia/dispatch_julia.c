#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <julia.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>
#include <oif/_platform.h>

static char *prefix_ = "dispatch_julia";

typedef struct {
    ImplInfo base;
    const char *module_name;
} JuliaImplInfo;

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
    char using_statement[1024];
    sprintf(using_statement, "using .%s", module_name);

    jl_value_t *retval;
    retval = jl_eval_string(include_statement);
    if (jl_exception_occurred()) {
        goto catch;
    }
    jl_static_show(jl_stdout_stream(), retval);
    jl_printf(jl_stdout_stream(), "\n");

    retval = jl_eval_string(using_statement);
    if (jl_exception_occurred()) {
        goto catch;
    }
    jl_static_show(jl_stdout_stream(), retval);
    jl_printf(jl_stdout_stream(), "\n");


    result = malloc(sizeof *result);
    if (result == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Julia implementation information\n",
                prefix_);
        goto cleanup;
    }
    result->module_name = module_name;

    fprintf(stderr,
            "[%s] I want to inform you that we successfully loaded stub of a Julia "
            "implementation!\n",
            prefix_);
    goto cleanup;

catch:
        // Handle the error
        jl_value_t *exception = jl_exception_occurred();
        // Print or handle the error as needed
        jl_printf(jl_stderr_stream(), "[%s] ", prefix_);
        jl_value_t *exception_str = jl_call1(jl_get_function(jl_base_module, "string"), exception);
        jl_printf(jl_stderr_stream(), "%s\n", jl_string_ptr(exception_str));

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
call_impl(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    (void)impl_info;
    (void)method;
    (void)in_args;
    (void)out_args;
    fprintf(stderr, "[%s] Stub for call_impl is invoked\n", prefix_);
    return 0;
}

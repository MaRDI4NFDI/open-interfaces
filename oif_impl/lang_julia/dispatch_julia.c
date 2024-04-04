#include <dlfcn.h>
#include <stdio.h>
#include <julia.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>

static char *prefix_ = "dispatch_julia";

typedef struct {
    ImplInfo base;
} JuliaImplInfo;


ImplInfo *
load_impl(const char *impl_details, size_t version_major, size_t version_minor) {
    (void) impl_details;
    (void)version_major;
    (void)version_minor;
    JuliaImplInfo *result = NULL;

    jl_init();

    result = malloc(sizeof *result);
    if (result == NULL) {
        fprintf(stderr, "[%s] Could not allocate memory for Julia implementation information\n", prefix_);
        goto cleanup;
    }

    fprintf(stderr, "[%s] I want to inform you that we successfully loaded stub of a Julia implementation!\n", prefix_);

cleanup:

    return (ImplInfo *)result;
}

int
unload_impl(ImplInfo *impl_info)
{
    (void)impl_info;
    fprintf(stderr, "[%s] Implement releasing memory for impl_info!!!!!\n", prefix_);
    jl_atexit_hook(0);
    return 0;
}

int call_impl(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    (void) impl_info;
    (void) method;
    (void) in_args;
    (void) out_args;
    fprintf(stderr, "[%s] Stub for call_impl is invoked\n", prefix_);
    return 0;
}

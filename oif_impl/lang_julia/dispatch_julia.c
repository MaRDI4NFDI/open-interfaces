#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <julia.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>
#include <oif/_platform.h>

static char *prefix_ = "dispatch_julia";

typedef struct {
    ImplInfo base;
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

    const char *module_filename = "oif_impl/impl/qeq/jl_qeq_solver/qeq_solver.jl";
    /* const char *module_name = "Main.QeqSolver"; */
    char include_statement[512];
    sprintf(include_statement, "include(\"%s\")", module_filename);
    jl_eval_string(include_statement);
    jl_eval_string("using .QeqSolver");
    /* jl_load(module, module_name); */
    /* jl_value_t * mod = (jl_value_t*)jl_eval_string("Solvers"); */
    /* jl_function_t * func = jl_get_function((jl_module_t*)module, "solve"); */

    int64_t (*func_jl)(double, double, double, double *) = jl_unbox_voidpointer(jl_eval_string("@cfunction(QeqSolver.solve!, Int, (Float64, Float64, Float64, Ref{Float64}))"));
    if (func_jl == NULL) {
        fprintf(
            stderr,
            "[%s] We could not obtain a pointer to C-compatible Julia function\n",
            prefix_
        );
    }
    double roots[2] = {99.0, 25.0};
    int ret = (int) func_jl(1.0, 5.0, 4.0, roots);
    fprintf(stderr, "[%s] We called QeqSolver.solve\n", prefix_);
    fprintf(stderr, "roots1 = %f, roots2 = %f\n", roots[0], roots[1]);

    result = malloc(sizeof *result);
    if (result == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Julia implementation information\n",
                prefix_);
        goto cleanup;
    }

    fprintf(stderr,
            "[%s] I want to inform you that we successfully loaded stub of a Julia "
            "implementation!\n",
            prefix_);

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

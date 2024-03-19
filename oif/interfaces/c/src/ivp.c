#include "oif/dispatch_api.h"
#include <stdio.h>
#include <stdlib.h>

#include <oif/api.h>
#include <oif/dispatch.h>
#include <oif/interfaces/ivp.h>

int
oif_ivp_set_rhs_fn(ImplHandle implh, oif_ivp_rhs_fn_t rhs)
{
    OIFCallback rhs_wrapper = {.src = OIF_LANG_C, .fn_p_py = NULL, .fn_p_c = rhs};
    OIFArgType in_arg_types[] = {OIF_CALLBACK};
    void *in_arg_values[] = {&rhs_wrapper};
    OIFArgs in_args = {
        .num_args = 1,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {};
    void *out_arg_values[] = {};
    OIFArgs out_args = {
        .num_args = 0,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_impl(implh, "set_rhs_fn", &in_args, &out_args);

    return status;
}

int
oif_ivp_set_initial_value(ImplHandle implh, OIFArrayF64 *y0, double t0)
{
    OIFArgType in_arg_types[] = {OIF_ARRAY_F64, OIF_FLOAT64};
    void *in_arg_values[] = {&y0, &t0};
    OIFArgs in_args = {
        .num_args = 2,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {};
    void *out_arg_values[] = {};
    OIFArgs out_args = {
        .num_args = 0,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_impl(implh, "set_initial_value", &in_args, &out_args);

    return status;
}

int oif_ivp_set_user_data(ImplHandle implh, void *user_data)
{
    OIFArgType in_arg_types[] = {OIF_VOID_P};
    void *in_arg_values[] = {&user_data};
    size_t n = sizeof(in_arg_values) / sizeof(in_arg_values[0]);
    OIFArgs in_args = {
        .num_args = n,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {};
    void *out_arg_values[] = {};
    size_t num_out_args = sizeof(out_arg_values) / sizeof(out_arg_values[0]);
    OIFArgs out_args = {
        .num_args = num_out_args,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_impl(implh, "set_user_data", &in_args, &out_args);
    return status;
}

int
oif_ivp_integrate(ImplHandle implh, double t, OIFArrayF64 *y)
{
    OIFArgType in_arg_types[] = {OIF_FLOAT64};
    void *in_arg_values[] = {&t};
    OIFArgs in_args = {
        .num_args = 1,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {OIF_ARRAY_F64};
    void *out_arg_values[] = {&y};
    OIFArgs out_args = {
        .num_args = 1,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_impl(implh, "integrate", &in_args, &out_args);

    return status;
}

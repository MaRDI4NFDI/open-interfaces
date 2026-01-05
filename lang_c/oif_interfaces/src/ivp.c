#include <stdlib.h>

#include <oif/api.h>
#include <oif/interfaces/ivp.h>
#include <oif/internal/dispatch.h>

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
    OIFArgType in_arg_types[] = {OIF_ARRAY_F64, OIF_TYPE_F64};
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

int
oif_ivp_set_user_data(ImplHandle implh, void *user_data)
{
    OIFArgType in_arg_types[] = {OIF_USER_DATA};
    OIFUserData oif_user_data = {
        .src = OIF_LANG_C,
        .c = user_data,
    };
    void *in_arg_values[] = {&oif_user_data};
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
oif_ivp_set_tolerances(ImplHandle implh, double rtol, double atol)
{
    OIFArgType in_arg_types[] = {OIF_TYPE_F64, OIF_TYPE_F64};
    void *in_arg_values[] = {&rtol, &atol};
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

    int status = call_interface_impl(implh, "set_tolerances", &in_args, &out_args);

    return status;
}

int
oif_ivp_integrate(ImplHandle implh, double t, OIFArrayF64 *y)
{
    OIFArgType in_arg_types[] = {OIF_TYPE_F64};
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

int
oif_ivp_set_integrator(ImplHandle implh, char *integrator_name, OIFConfigDict *dict)
{
    if (dict != NULL) {
        oif_config_dict_serialize(dict);
    }
    OIFArgType in_arg_types[] = {OIF_STR, OIF_CONFIG_DICT};
    void *in_arg_values[] = {&integrator_name, &dict};
    size_t in_num_args = sizeof(in_arg_types) / sizeof(*in_arg_types);
    OIFArgs in_args = {
        .num_args = in_num_args,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {};
    void *out_arg_values[] = {};
    size_t out_num_args = sizeof(out_arg_values) / sizeof(*out_arg_values);
    OIFArgs out_args = {
        .num_args = out_num_args,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_impl(implh, "set_integrator", &in_args, &out_args);
    return status;
}

int
oif_ivp_print_stats(ImplHandle implh)
{
    OIFArgs in_args = {
        .num_args = 0,
        .arg_types = NULL,
        .arg_values = NULL,
    };

    OIFArgs out_args = {
        .num_args = 0,
        .arg_types = NULL,
        .arg_values = NULL,
    };

    int status = call_interface_impl(implh, "print_stats", &in_args, &out_args);
    return status;
}

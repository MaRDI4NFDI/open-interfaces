#include <stdlib.h>

#include <oif/api.h>
#include <oif/dispatch.h>
#include <oif/interfaces/qeq.h>

int oif_solve_qeq(
    ImplHandle implh, double a, double b, double c, OIFArrayF64 *roots) {
    OIFArgType in_arg_types[3] = {OIF_FLOAT64, OIF_FLOAT64, OIF_FLOAT64};
    void *in_arg_values[3] = {(void *)&a, (void *)&b, (void *)&c};
    OIFArgs in_args = {
        .num_args = 3,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {OIF_ARRAY_F64};
    void *out_arg_values[] = {&roots};
    OIFArgs out_args = {
        .num_args = 1,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_method(implh, "solve_qeq", &in_args, &out_args);

    return status;
}

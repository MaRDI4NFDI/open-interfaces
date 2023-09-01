#include "oif/api.h"
#include <oif/frontend_c/qeq.h>
#include <dispatch.h>
#include <stdlib.h>

int oif_solve_qeq(
    BackendHandle bh, double a, double b, double c, OIFArray *roots
) {
    OIFArgType in_arg_types[3] = {OIF_FLOAT64, OIF_FLOAT64, OIF_FLOAT64};
    void *in_arg_values[3] = {(void *)&a, (void *)&b, (void *)&c};
    OIFArgs in_args = {
        .num_args = 3,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {OIF_FLOAT64_P};
    void *out_arg_values[] = {&roots};
    OIFArgs out_args = {
        .num_args = 1,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_method(
        bh,
        "solve_qeq",
        &in_args,
        &out_args
    );

    return status;
}


#include <stdlib.h>

#include <oif/api.h>
#include <oif/dispatch.h>
#include <oif/interfaces/linsolve.h>

int oif_solve_linear_system(
    BackendHandle bh, OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x
) {
    OIFArgType in_arg_types[] = {OIF_ARRAY_F64, OIF_ARRAY_F64};
    void *in_arg_values[] = {(void *)&A, (void *)&b};
    OIFArgs in_args = {
        .num_args = 2,
        .arg_types = in_arg_types,
        .arg_values = in_arg_values,
    };

    OIFArgType out_arg_types[] = {OIF_ARRAY_F64};
    void *out_arg_values[] = {(void *) &x};
    OIFArgs out_args = {
        .num_args = 1,
        .arg_types = out_arg_types,
        .arg_values = out_arg_values,
    };

    int status = call_interface_method(
        bh,
        "solve_lin",
        &in_args,
        &out_args
    );

    return status;
}


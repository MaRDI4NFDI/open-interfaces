#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>


char *parse_impl(int argc, char *argv[]) {
    if (argc == 1) {
        return "scipy_ode_dopri5";
    } else {
        if ((strcmp(argv[1], "scipy_ode_dopri5") == 0)) {
            return argv[1];
        } else {
            fprintf(stderr, "USAGE: %s [scipy_ode_dopri5]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

void rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out) {
    int size = y->dimensions[0];
    for (int i = 0; i < size; ++i) {
        rhs_out->data[i] = -y->data[i];
    }
}

int main(int argc, char *argv[])
{
    char *impl = parse_impl(argc, argv);
    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("Implementation: %s\n", impl);

    double t0 = 0.0;
    OIFArrayF64 *y0 = oif_init_array_f64_from_data(
        1, (intptr_t [1]){1}, (double [1]){1.0}
    );

    ImplHandle implh = oif_init_impl("ivp", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    int status;  // Aux variable to check for errors.
    status = oif_ivp_set_rhs_fn(implh, rhs);
    if (status) {
        fprintf(stderr, "oif_ivp_set_rhs_fn returned error\n");
        return EXIT_FAILURE;
    }
    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status) {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        return EXIT_FAILURE;
    }

    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t [1]){1});
    // Time step.
    double dt = 0.1;
    for (double t = t0 + dt; t <= 1.0; t += dt) {
        status = oif_ivp_integrate(implh, t, y);
        if (status) {
            fprintf(stderr, "oif_ivp_integrate returned error\n");
            return EXIT_FAILURE;
        }
        printf("%.3f %.6f\n", t, y->data[0]);
    }

    return 0;
}

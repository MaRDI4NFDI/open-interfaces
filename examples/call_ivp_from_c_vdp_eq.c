/**
 * To plot the solution, use the following Python code:
 * import matplotlib.pyplot as plt
 * import numpy as np
 *
 * t, u = np.loadtxt("solution.txt")
 *
 * plt.figure()
 * plt.plot(t, u, "-", label=r"y(t)")
 * plt.show()
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/config_dict.h>
#include <oif/interfaces/ivp.h>

// Number of right-hand side evaluations.
static int N_RHS_EVALS = 0;

char *
parse_impl(int argc, char *argv[])
{
    if (argc == 1) {
        return "scipy_ode";
    }
    else {
        if ((strcmp(argv[1], "scipy_ode") == 0) || (strcmp(argv[1], "sundials_cvode") == 0) ||
            (strcmp(argv[1], "jl_diffeq") == 0)) {
            return argv[1];
        }
        else {
            fprintf(stderr, "USAGE: %s [scipy_ode | sundials_cvode | jl_diffeq]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

char *
parse_integrator(int argc, char *argv[])
{
    return argv[2];
}

char *
parse_output_filename(int argc, char *argv[])
{
    if (argc < 4) {
        return "solution.txt";
    }
    else {
        return argv[3];
    }
}

int
rhs(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data)
{
    (void)t;  // Unused
    double mu = *(double *)user_data;
    ydot->data[0] = y->data[1];
    ydot->data[1] = mu * (1 - y->data[0] * y->data[0]) * y->data[1] - y->data[0];

    return 0;
}

int
main(int argc, char *argv[])
{
    char *impl = parse_impl(argc, argv);
    char *integrator = parse_integrator(argc, argv);
    const char *output_filename = parse_output_filename(argc, argv);
    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("where the system comes from the 2nd-order Van der Pol oscillator equation\n");
    printf("Implementation: %s\n", impl);
    printf("Integrator: %s\n", integrator);


    int retval = 0;

    const int N = 2;  // Number of equations in the system.

    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N});

    double t0 = 0.0;
    double t_final = 3000;
    double mu = 1e3; // Stiffness parameter.
    y0->data[0] = 2.0;
    y0->data[1] = 0.0;

    int status = 1;  // Aux variable to check for errors.

    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status) {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_user_data(implh, &mu);
    if (status) {
        fprintf(stderr, "oif_ivp_set_user_data return error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_rhs_fn(implh, rhs);
    if (status) {
        fprintf(stderr, "oif_ivp_set_rhs_fn returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }

    status = oif_ivp_set_tolerances(implh, 1e-8, 1e-12);
    assert(status == 0);

    OIFConfigDict *dict = oif_config_dict_init();

    if (strcmp(impl, "sundials_cvode") == 0) {
        oif_config_dict_add_int(dict, "max_num_steps", 30000);
        status = oif_ivp_set_integrator(implh, "bdf", dict);
    }
    else if (strcmp(impl, "scipy_ode") == 0 && strcmp(integrator, "dopri5") == 0) {
        // It is already the default integrator.
        status = oif_ivp_set_integrator(implh, "dopri5", NULL);
    }
    else if (strcmp(impl, "scipy_ode") == 0 && strcmp(integrator, "dopri5-100k") == 0) {
        oif_config_dict_add_int(dict, "nsteps", 100000);
        status = oif_ivp_set_integrator(implh, "dopri5", dict);
    }
    else if (strcmp(impl, "scipy_ode") == 0 && strcmp(integrator, "vode") == 0) {
        oif_config_dict_add_str(dict, "method", "bdf");
        status = oif_ivp_set_integrator(implh, "vode", dict);
    }
    else if (strcmp(impl, "scipy_ode") == 0 && strcmp(integrator, "vode-40k") == 0) {
        printf("I AM HERE\n");
        oif_config_dict_add_str(dict, "method", "bdf");
        oif_config_dict_add_int(dict, "nsteps", 40000);
        status = oif_ivp_set_integrator(implh, "vode", dict);
    }
    else if (strcmp(impl, "jl_diffeq") == 0 && strcmp(integrator, "rosenbrock23") == 0) {
        oif_config_dict_add_int(dict, "autodiff", 0);
        status = oif_ivp_set_integrator(implh, "Rosenbrock23", dict);
    }
    else {
        fprintf(stderr, "Cannot set integrator for implementation '%s'\n", impl);
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    assert(status == 0);

    double t = 0.0001;
    status = oif_ivp_integrate(implh, t, y);

    const int Nt = 501; // Number of time steps.
    double dt = (t_final - t0) / (Nt - 1);

    OIFArrayF64 *times = oif_create_array_f64(1, (intptr_t[1]){Nt});
    OIFArrayF64 *solution = oif_create_array_f64(1, (intptr_t[1]){Nt});
    times->data[0] = t0;
    solution->data[0] = y0->data[0];

    clock_t tic = clock();
    // Time step.
    for (int i = 1; i < Nt; ++i) {
        double t = t0 + i * dt;
        if (t > t_final) {
            t = t_final;
        }
        status = oif_ivp_integrate(implh, t, y);
        if (status) {
            fprintf(stderr, "oif_ivp_integrate returned error\n");
            retval = EXIT_FAILURE;
            goto cleanup;
        }
        times->data[i] = t;
        solution->data[i] = y->data[0];
    }
    clock_t toc = clock();
    printf("Elapsed time = %.6f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    printf("Number of right-hand side evaluations = %d\n", N_RHS_EVALS);

    FILE *fp = fopen(output_filename, "w+e");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s' for writing\n", output_filename);
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    for (int i = 0; i < Nt; ++i) {
        fprintf(fp, "%.8f %.8f\n", times->data[i], solution->data[i]);
    }
    fclose(fp);
    printf("Solution was written to file `%s`\n", output_filename);

cleanup:
    oif_free_array_f64(solution);
    oif_free_array_f64(times);
    oif_free_array_f64(y);
    oif_free_array_f64(y0);
    oif_config_dict_free(dict);

    oif_unload_impl(implh);

    return retval;
}

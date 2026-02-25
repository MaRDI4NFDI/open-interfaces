/**
 * To plot the solution, use the following Python code:
 * import matplotlib.pyplot as plt
 * import numpy as np
 *
 * u = np.loadtxt("solution.txt")
 *
 * plt.figure()
 * plt.plot(u, "-")
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
parse_output_filename(int argc, char *argv[])
{
    if (argc == 1 || argc == 2) {
        return "solution.txt";
    }
    else {
        return argv[2];
    }
}

int
parse_resolution(int argc, char *argv[])
{
    printf("argc = %d\n", argc);
    if (argc <= 3) {
        /* return 101; */
        /* TODO: return 101!!! */
        return 10;
    }
    else {
        return atoi(argv[3]);
    }
}

static int
compute_initial_condition_(size_t N, OIFArrayF64 *u0, OIFArrayF64 *grid, double *dx,
                           double *dt_max)
{
    double a = 0.0;
    double b = 2.0;
    double *x = grid->data;
    *dx = (b - a) / N;

    for (int i = 0; i < N; ++i) {
        x[i] = a + i * (*dx);
    }

    for (int i = 0; i < N; ++i) {
        u0->data[i] = 0.5 - 0.25 * sin(M_PI * x[i]);
    }

    double cfl = 0.5;
    *dt_max = cfl * (*dx);

    return 0;
}

int
rhs(double t, OIFArrayF64 *y, OIFArrayF64 *rhs_out, void *user_data)
{
    (void)t; /* Unused */
    int retval = 1;
    intptr_t N = y->dimensions[0];
    assert(N > 1);

    double *u = y->data;
    double *udot = rhs_out->data;

    // User data is just the spatial step size `dx`.
    double dx = *((double *)user_data);

    double *flux = malloc(N * sizeof(double));
    if (flux == NULL) {
        fprintf(stderr, "Could not allocate memory for the flux array\n");
        return retval;
    }

    for (int i = 0; i < N; ++i) {
        flux[i] = 0.5 * pow(u[i], 2.0);
    }

    double local_sound_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        if (local_sound_speed < fabs(u[i])) {
            local_sound_speed = fabs(u[i]);
        }
    }

    double *flux_hat = malloc((N - 1) * sizeof(double));
    if (flux_hat == NULL) {
        fprintf(stderr, "Could not allocate memory for flux_hat\n");
        retval = 1;
        goto cleanup;
    }

    for (int i = 0; i < N - 1; ++i) {
        flux_hat[i] =
            0.5 * (flux[i] + flux[i + 1]) - 0.5 * local_sound_speed * (u[i + 1] - u[i]);
    }

    for (int i = 1; i < N - 1; ++i) {
        udot[i] = -1.0 / dx * (flux_hat[i] - flux_hat[i - 1]);
    }
    double f_rb = 0.5 * (flux[0] + flux[N - 1]) - 0.5 * local_sound_speed * (u[0] - u[N - 1]);
    double f_lb = f_rb;
    udot[0] = -1.0 / dx * (flux_hat[0] - f_lb);
    udot[N - 1] = -1.0 / dx * (f_rb - flux_hat[N - 2]);

    retval = 0;

    N_RHS_EVALS++;

cleanup:
    if (flux != NULL) {
        free(flux);
    }
    if (flux_hat != NULL) {
        free(flux_hat);
    }

    return retval;
}

int
main(int argc, char *argv[])
{
    char *impl = parse_impl(argc, argv);
    const char *output_filename = parse_output_filename(argc, argv);
    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("where the system comes from inviscid 1D Burgers' equation\n");
    printf("Implementation: %s\n", impl);
    printf("Output filename: %s\n", output_filename);

    int retval = 0;

    ImplHandle implh = oif_load_impl("ivp", impl, 1, 0);
    if (implh < OIF_IMPL_STARTING_NUMBER) {
        goto fail;
    }

    const int N = parse_resolution(argc, argv);
    printf("N = %d\n", N);
    double t0 = 0.0;
    double t_final = 2.0;
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});
    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N});
    // Grid
    OIFArrayF64 *grid = oif_create_array_f64(1, (intptr_t[1]){N});
    double dx;
    double dt_max;
    int status = 1;  // Aux variable to check for errors.
    OIFConfigDict *dict = NULL;

    status = compute_initial_condition_(N, y0, grid, &dx, &dt_max);
    assert(status == 0);
    printf("dx = %.16f\n", dx);
    printf("dt_max = %.16f\n", dt_max);

    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status) {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        retval = EXIT_FAILURE;
        goto cleanup;
    }
    status = oif_ivp_set_user_data(implh, &dx);
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

    status = oif_ivp_set_tolerances(implh, 1e-2, 1e-2);
    assert(status == 0);

    dict = oif_config_dict_init();
    oif_config_dict_add_int(dict, "dense", 0);
    oif_config_dict_add_int(dict, "save_everystep", 0);

    if (strcmp(impl, "scipy_ode") == 0) {
        status = oif_ivp_set_integrator(implh, "dopri5", NULL);
    }
    else if (strcmp(impl, "jl_diffeq") == 0) {
        status = oif_ivp_set_integrator(implh, "DP5", NULL);
        /* status = oif_ivp_set_integrator(implh, "DP5", dict); */
    }
    assert(status == 0);
    double t = 0.0001;
    status = oif_ivp_integrate(implh, t, y);
    assert(status == 0);

    clock_t tic = clock();
    // Time step.
    double dt = dt_max;
    int n_time_steps = (int)(t_final / dt + 1);
    for (int i = 0; i < n_time_steps; ++i) {
        double t = t0 + (i + 1) * dt;
        if (t > t_final) {
            t = t_final;
        }
        char const *sep = "";
        printf("Timestep %d, time %.16f\n", i, t);
        for (int k = 0; k < N; k++) {
            printf("%s%2d:%.3f", sep, k, y->data[k]);
            sep = " ";
            if ((k + 1) % 10 == 0) {
                printf("\n");
                sep = "";
            }
        }
        printf("\n");
        status = oif_ivp_integrate(implh, t, y);
        if (status) {
            fprintf(stderr,
                    "oif_ivp_integrate returned error when integrating to time %.16f, "
                    "timestep %d\n",
                    t, i);
            retval = EXIT_FAILURE;
            goto cleanup;
        }
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
    for (int i = 0; i < N; ++i) {
        fprintf(fp, "%.8f %.8f\n", grid->data[i], y->data[i]);
    }
    fclose(fp);
    printf("Solution was written to file `%s`\n", output_filename);

cleanup:
    oif_free_array_f64(y0);
    oif_free_array_f64(y);
    oif_free_array_f64(grid);

    oif_config_dict_free(dict);
    oif_unload_impl(implh);

    return retval;

fail:
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }
    if (implh == OIF_BRIDGE_NOT_AVAILABLE_ERROR) {
        fprintf(stderr,
                "Bridge component for the implementation '%s' is not available. "
                "Cannot proceed\n",
                impl);
        return OIF_BRIDGE_NOT_AVAILABLE_ERROR;
    }
    if (implh == OIF_IMPL_NOT_AVAILABLE_ERROR) {
        fprintf(stderr, "Implementation '%s' is not available. Cannot proceed\n", impl);
        return OIF_IMPL_NOT_AVAILABLE_ERROR;
    }
}

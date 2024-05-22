#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/interfaces/ivp.h>

char *
parse_impl(int argc, char *argv[])
{
    if (argc == 1) {
        return "scipy_ode_dopri5";
    }
    else {
        if ((strcmp(argv[1], "scipy_ode_dopri5") == 0) ||
            (strcmp(argv[1], "sundials_cvode") == 0) || (strcmp(argv[1], "jl_diffeq") == 0)) {
            return argv[1];
        }
        else {
            fprintf(stderr, "USAGE: %s [scipy_ode_dopri5 | sundials_cvode | jl_diffeq]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

static int
compute_initial_condition_(size_t N, OIFArrayF64 *u0, double *dx, double *dt_max)
{
    double a = 0.0;
    double b = 2.0;
    double *x = malloc(N * sizeof(double));
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
    (void)t;         /* Unused */
    (void)user_data; /* Unused */
    intptr_t N = y->dimensions[0];

    double *u = y->data;
    double *udot = rhs_out->data;

    double dx = *((double *) user_data);

    double *f = malloc(N * sizeof(double));
    if (f == NULL) {
        fprintf(stderr, "Could not allocate memory for the flux array\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        f[i] = 0.5 * pow(u[i], 2.0);
    }

    double local_sound_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        if (local_sound_speed < fabs(u[i])) {
            local_sound_speed = fabs(u[i]);
        }
    }

    double *f_hat = malloc((N - 1) * sizeof(double));
    if (f_hat == NULL) {
        fprintf(stderr, "Could not allocate memory for f_hat\n");
        return 1;
    }

    for (int i = 0; i < N - 1; ++i) {
        f_hat[i] = 0.5 * (f[i] + f[i + 1]) -
                   0.5 * local_sound_speed * (u[i + 1] - u[i]);
    }

    for (int i = 1; i < N - 1; ++i) {
        udot[i] = -1.0 / dx * (f_hat[i] - f_hat[i - 1]);
    }
    double f_rb = 0.5 * (f[0] + f[N-1]) -
        0.5 * local_sound_speed * (u[0] - u[N-1]);
    double f_lb = f_rb;
    udot[0] = -1.0 / dx * (f_hat[0] - f_lb);
    udot[N - 1] = -1.0 / dx * (f_rb - f_hat[N-2]);

    return 0;
}

int
main(int argc, char *argv[])
{
    char *impl = parse_impl(argc, argv);
    printf("Calling from C an open interface for solving y'(t) = f(t, y)\n");
    printf("where the system comes from inviscid 1D Burgers' equation\n");
    printf("Implementation: %s\n", impl);

    int N = 101;
    double t0 = 0.0;
    double t_final = 2.0;
    OIFArrayF64 *y0 = oif_create_array_f64(1, (intptr_t[1]){N});

    double dx;
    double dt_max;
    int status = 1;  // Aux variable to check for errors.

    status = compute_initial_condition_(N, y0, &dx, &dt_max);
    assert(status == 0);

    ImplHandle implh = oif_init_impl("ivp", impl, 1, 0);
    if (implh == OIF_IMPL_INIT_ERROR) {
        fprintf(stderr, "Error during implementation initialization. Cannot proceed\n");
        return EXIT_FAILURE;
    }

    status = oif_ivp_set_initial_value(implh, y0, t0);
    if (status) {
        fprintf(stderr, "oif_ivp_set_set_initial_value returned error\n");
        return EXIT_FAILURE;
    }
    status = oif_ivp_set_rhs_fn(implh, rhs);
    if (status) {
        fprintf(stderr, "oif_ivp_set_rhs_fn returned error\n");
        return EXIT_FAILURE;
    }

    status = oif_ivp_set_user_data(implh, &dx);
    if (status) {
        fprintf(stderr, "oif_ivp_set_user_data return error\n");
        return EXIT_FAILURE;
    }

    // Solution vector.
    OIFArrayF64 *y = oif_create_array_f64(1, (intptr_t[1]){N});
    // Time step.
    double dt = dt_max;
    int n_time_steps = (int) (t_final / dt + 1);
    for (int i = 0; i < n_time_steps; ++i) {
        printf("%d\n", i);
        double t = t0 + (i + 1) * dt;
        if (t > t_final) {
            t = t_final;
        }
        status = oif_ivp_integrate(implh, t, y);
        if (status) {
            oif_free_array_f64(y0);
            oif_free_array_f64(y);
            fprintf(stderr, "oif_ivp_integrate returned error\n");
            return EXIT_FAILURE;
        }
    }

    FILE *fp = fopen("solution.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open file for writing\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        fprintf(fp, "%.8f\n", y->data[i]);
    }
    fclose(fp);

    oif_free_array_f64(y0);
    oif_free_array_f64(y);

    return 0;
}


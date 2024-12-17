/**
 * Implementation of the `ivp` interface with hand-written Dormand-Prince 5(4).
 */
#include <assert.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#include <oif/api.h>
#include <oif/util.h>
#include <oif_impl/ivp.h>
#include <oif/c_bindings.h>

#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

static const char *prefix_ = "ivp::dopri5c";

double self_t = 0.0;
double self_rtol = 1e-6;   // relative tolerance
double self_atol = 1e-12;  // absolute tolerance
int self_N = 0;            // Length of the solution vector.
OIFArrayF64 *self_y = NULL;
OIFArrayF64 *self_yt = NULL;

// Runge--Kutta stages.
OIFArrayF64 *self_k1 = NULL;
OIFArrayF64 *self_k2 = NULL;
OIFArrayF64 *self_k3 = NULL;
OIFArrayF64 *self_k4 = NULL;
OIFArrayF64 *self_k5 = NULL;
OIFArrayF64 *self_k6 = NULL;

OIFArrayF64 *self_y2 = NULL;
OIFArrayF64 *self_y3 = NULL;
OIFArrayF64 *self_y4 = NULL;
OIFArrayF64 *self_y5 = NULL;
OIFArrayF64 *self_y6 = NULL;

OIFArrayF64 *self_sc = NULL;

OIFConfigDict *config = NULL;
void *self_user_data = NULL;
// Signature for the right-hand side that is provided by the `IVP` interface.
static oif_ivp_rhs_fn_t OIF_RHS_FN = NULL;

static double self_h = 0.0;

static const char *AVAILABLE_OPTIONS_[] = {
    "max_num_steps",
    NULL,
};

static double eps = 1e-12;

// Leading zeros here are only to match the index to the corresponding vector.
static double a4[] = {0, 44.0L/45, -56.0L/15, 32.0L/9};
static double a5[] = {0, 19372.0L/6561, -25360.0L/2187, 64448.0L/6561, -212.0L/729};
static double a6[] = {0, 9017.0L/3168, -355.0L/33, 46732.0L/5247, 49.0L/176, -5103.0L/18656};
static double a7[] = {0, 35.0L/384, 0.0, 500.0L/1113, 125.0/192, -2187.0/6784, 11.0/84};
static double e[] = {0, 71.0L/57600, -1.0/40, -71.0L/16695, 71.0L/1920, -17253.0L/339200, 22.0L/525};

// Step size control parameters.
double SAFE = 0.9L;
double BETA = 0.04L;
double EXP01 = 0.0;
double FAC1 = 0.2L;
double FACC1;

double const FACMAX_DEFAULT = 1.5;
double const FACMIN = 0.2;
double const FAC = 0.8;
double const ORDER_OF_ACC = 4;
double FACMAX;  // FACMAX depends on step acceptance/rejection (see Hairer, vol. 1, p. 168)

size_t n_rejected = 0;


static int
init_(void)
{
    return 0;
}

static void
compute_initial_step_() {
    double h0;
    double h1;

    OIF_RHS_FN(self_t, self_y, self_k1, self_user_data);

    for (int i = 0; i < self_N; ++i) {
        self_sc->data[i] = self_atol + self_y->data[i] * self_rtol;
    }

    double d0 = 0.0L;
    for (int i = 0; i < self_N; ++i) {
        d0 += pow(self_y->data[i] / self_sc->data[i], 2);
    }
    d0 = sqrt(d0 / self_N);

    double d1 = 0.0L;
    for (int i = 0; i < self_N; ++i) {
        d1 += pow(self_k1->data[i] / self_sc->data[i], 2);
    }
    d1 = sqrt(d1 / self_N);

    if (d0 < 1e-5 || d1 < 1e-5) {
        h0 = 1e-6;
    }
    else {
        h0 = 1e-2 * (d0 / d1);
    }

    for (int i = 0; i < self_N; ++i) {
        self_y2->data[i] = self_y->data[i] + h0 * self_k1->data[i];
    }
    OIF_RHS_FN(self_t + self_h, self_y2, self_k2, self_user_data);

    double d2 = 0.0L;
    for (int i = 0; i < self_N; ++i) {
        d2 += pow((self_k2->data[i] - self_k1->data[i]) / self_sc->data[i], 2.0);
    }
    d2 = sqrt(d2 / self_N) / h0;

    if (MAX(d1, d2) < 1e-15) {
        h1 = MAX(1e-6, h0 * 1e-3);
    }
    else {
        h1 = pow(1e-2 / MAX(d1, d2), 1.0L / 5.0L);
    }

    self_h = MIN(1e2 * h0, h1);
}

int
set_initial_value(OIFArrayF64 *y0_in, double t0_in)
{
    EXP01 = 0.2L - BETA * 0.75L;
    FACC1 = 1.0L / FAC1;
    FACMAX = FACMAX_DEFAULT;
    self_t = t0_in;

    if (y0_in->nd != 1) {
        fprintf(
            stderr,
            "[%s] Accept only one-dimensional arrays (vectors)\n",
            prefix_
        );
    }

    if (self_y) {
        oif_free_array_f64(self_y);
        oif_free_array_f64(self_k1);
        oif_free_array_f64(self_k2);
        oif_free_array_f64(self_k3);
        oif_free_array_f64(self_k4);
        oif_free_array_f64(self_k5);
        oif_free_array_f64(self_k6);
    }

    self_N = y0_in->dimensions[0];

    self_y = oif_create_array_f64(1, y0_in->dimensions);
    if (self_y == NULL) {
        fprintf(stderr, "[%s] Could not allocate memory for the solution vector\n", prefix_);
        exit(1);
    }
    for (size_t i = 0; i < self_y->dimensions[0]; ++i) {
        self_y->data[i] = y0_in->data[i];
    }

    self_yt = oif_create_array_f64(1, y0_in->dimensions);

    self_k1 = oif_create_array_f64(1, y0_in->dimensions);
    self_k2 = oif_create_array_f64(1, y0_in->dimensions);
    self_k3 = oif_create_array_f64(1, y0_in->dimensions);
    self_k4 = oif_create_array_f64(1, y0_in->dimensions);
    self_k5 = oif_create_array_f64(1, y0_in->dimensions);
    self_k6 = oif_create_array_f64(1, y0_in->dimensions);

    self_y2 = oif_create_array_f64(1, y0_in->dimensions);
    self_y3 = oif_create_array_f64(1, y0_in->dimensions);
    self_y4 = oif_create_array_f64(1, y0_in->dimensions);
    self_y5 = oif_create_array_f64(1, y0_in->dimensions);
    self_y6 = oif_create_array_f64(1, y0_in->dimensions);

    self_sc = oif_create_array_f64(1, y0_in->dimensions);

    return 0;
}

int
set_user_data(void *user_data)
{
    self_user_data = user_data;
    return 0;
}

int
set_rhs_fn(oif_ivp_rhs_fn_t rhs)
{
    if (rhs == NULL) {
        fprintf(stderr, "[%s] `set_rhs_fn` accepts non-null function pointer only\n", prefix_);
        return 1;
    }
    OIF_RHS_FN = rhs;
    compute_initial_step_();
    return 0;
}

int
set_tolerances(double rtol, double atol)
{
    self_rtol = rtol;
    self_atol = atol;
    return 0;
}

int
set_integrator(const char *integrator_name, OIFConfigDict *config_)
{
    (void) integrator_name;
    (void) config_;
    fprintf(stderr, "[%s] Method `set_integrator` is not supported\n", prefix_);
    return 0;
}

int
print_stats(void)
{
    return 0;
}

/**
 * Write into `y` the updated solution at time `t`.
 *
 */
int
integrate(double t_, OIFArrayF64 *y)
{
    fprintf(stderr, "[%s] integrate", prefix_);
    if (t_ <= self_t) {
        fprintf(stderr, "[%s] Time should be larger than the current time\n", prefix_);
    }

    while (self_t < t_) {
        if (self_t + self_h > t_) {
            self_h = t_ - self_t;
        }

        // 1st stage
        OIF_RHS_FN(self_t, y, self_k1, self_user_data);

        // 2nd stage
        for (int i = 0; i < self_N; ++i) {
            self_y2->data[i] = self_y->data[i] + self_h / 5.0 * self_k1->data[i];
        }
        OIF_RHS_FN(self_t + self_h / 5.0, self_y2, self_k2, self_user_data);

        // 3rd stage
        for (int i = 0; i < self_N; ++i) {
            self_y3->data[i] = self_y->data[i] + self_h / 40.0 * (
                3 * self_k1->data[i] + 9 * self_k2->data[i]
            );
        }
        OIF_RHS_FN(self_t + 3.0 * self_h / 10, self_y3, self_k3, self_user_data);

        // 4th stage
        for (int i = 0; i < self_N; ++i) {
            self_y4->data[i] = self_y->data[i] + self_h * (
                a4[1] * self_k1->data[i] +
                a4[2] * self_k2->data[i] +
                a4[3] * self_k3->data[i]
            );
        }
        OIF_RHS_FN(self_t + 4.0 * self_h / 5, self_y4, self_k4, self_user_data);

        // 5th stage
        for (int i = 0; i < self_N; ++i) {
            self_y5->data[i] = self_y->data[i] + self_h * (
                a5[1] * self_k1->data[i] +
                a5[2] * self_k2->data[i] +
                a5[3] * self_k3->data[i] +
                a5[4] * self_k4->data[i]
            );
        }
        OIF_RHS_FN(self_t + 8.0 * self_h / 9, self_y5, self_k5, self_user_data);

        // step 6.
        for (int i = 0; i < self_N; ++i) {
            self_y6->data[i] = self_y->data[i] + self_h * (
                a6[1] * self_k1->data[i] +
                a6[2] * self_k2->data[i] +
                a6[3] * self_k3->data[i] +
                a6[4] * self_k4->data[i] +
                a6[5] * self_k5->data[i]
            );
        }
        OIF_RHS_FN(self_t + self_h, self_y6, self_k6, self_user_data);

        // Estimate less accurate.
        for (size_t i = 0; i < self_y->dimensions[0]; ++i) {
            self_yt->data[i] = y->data[i] + self_h * (
                a7[1] * self_k1->data[i] + a7[3] * self_k3->data[i] +
                a7[4] * self_k4->data[i] + a7[5] * self_k5->data[i] +
                a7[6] * self_k6->data[i]
            );
        }
        OIF_RHS_FN(self_t + self_h, self_yt, self_k2, self_user_data);

        double err = 0.0;

        for (size_t i = 0; i < self_N; ++i) {
            double var = self_h * (
                e[1] * self_k1->data[i] + e[2] * self_k2->data[i] +
                e[3] * self_k3->data[i] + e[4] * self_k4->data[i] +
                e[5] * self_k5->data[i] + e[6] * self_k6->data[i]
            );
            double sk = self_atol + self_rtol * MAX(fabs(self_y->data[i]), fabs(self_yt->data[i]));
            err += pow(var / sk, 2);
        }
        err = sqrt(err / self_N);

        // double FAC11 = err * EXP01;
        double hnew = self_h * MIN(FACMAX, MAX(FACMIN, FAC * pow(1.0 / err, 1 / (ORDER_OF_ACC + 1.0))));
        if (err < 1.0) {
            // Solution accepted.
            self_t += self_h;
            OIFArrayF64 *tmp = self_k1;
            self_k1 = self_k2;
            self_k2 = tmp;

            tmp = self_y;
            self_y = self_yt;
            self_yt = tmp;
            FACMAX = FACMAX_DEFAULT;
            fprintf(stderr, "[%s::integrate] step is accepted\n", prefix_);
            n_rejected = 0;
        }
        else {
            // Solution rejected.
            // It is also advisable to put facmax = 1 in the steps
            // right after a step-rejection (Shampine & Watts 1979).
            // Hairer et. al., vol. 1, p. 168.
            FACMAX = 1.0;
            fprintf(stderr, "[%s::integrate] step is rejected\n", prefix_);
            n_rejected++;
        }

        self_h = hnew;
        fprintf(stderr, "[%s::integrate] h = %.16f\n", prefix_, h);

        if (n_rejected > 20) {
            fprintf(stderr, "[%s::integrate] Too many rejected steps\n", prefix_);
            exit(1);
        }
    }

    return 0;
}

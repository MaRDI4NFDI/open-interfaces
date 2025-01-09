/**
 * Implementation of the `ivp` interface with hand-written Dormand-Prince 5(4).
 *
 * Adaptive time step algorithm, computation of initial time step, and
 * related constants are from
 * Hairer et al. Solving Ordinary Differential Equations I, p. 167--169.
 *
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

// Runge--Kutta stages.
OIFArrayF64 *self_k1 = NULL;
OIFArrayF64 *self_k2 = NULL;
OIFArrayF64 *self_k3 = NULL;
OIFArrayF64 *self_k4 = NULL;
OIFArrayF64 *self_k5 = NULL;
OIFArrayF64 *self_k6 = NULL;

OIFArrayF64 *self_y = NULL;
OIFArrayF64 *self_y1 = NULL;
OIFArrayF64 *self_ysti = NULL;

OIFArrayF64 *self_sc = NULL;

OIFConfigDict *config = NULL;
void *self_user_data = NULL;
// Signature for the right-hand side that is provided by the `IVP` interface.
static oif_ivp_rhs_fn_t OIF_RHS_FN = NULL;

static double self_h = 0.0;

// Number of right-hand side function evaluations.
static size_t nfcn_ = 0;

// Coefficients before time step in expressions like t + c * dt.
static double C2 = 1.0 / 5.0;
static double C3 = 3.0 / 10.0;
static double C4 = 4.0 / 5.0;
static double C5 = 8.0 / 9.0;

// Leading zeros here are only to match the index to the corresponding vector.
static double a2[] = {0.0, 0.2};
static double a3[] = {0.0, 3.0 / 40.0, 9.0 / 40.0};
static double a4[] = {0, 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0};
static double a5[] = {0, 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0,
                      -212.0 / 729.0};
static double a6[] = {
    0, 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0};
static double a7[] = {
    0, 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0};
static double e[] = {
    0.0,           71.0 / 57600.0,      0.0,          -71.0 / 16695.0,
    71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0,
};

// Step size control parameters.
double SAFE = 0.9;
double BETA = 0.04;
double EXP01 = 0.0;
double FACC1;
double FACC2;

double FACOLD = 1e-4;
double const FACMIN = 0.2;   // In Hairer's code FAC1
double const FACMAX = 10.0;  // In Hairer's code FAC2
double FAC = 0.8;
double const ORDER_OF_ACC = 4;

size_t n_rejected = 0;

static void
compute_initial_step_()
{
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
        self_y1->data[i] = self_y->data[i] + h0 * self_k1->data[i];
    }
    OIF_RHS_FN(self_t + self_h, self_y1, self_k2, self_user_data);

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
    FACC1 = 1.0L / FACMIN;
    self_t = t0_in;

    if (y0_in->nd != 1) {
        fprintf(stderr, "[%s] Accept only one-dimensional arrays (vectors)\n", prefix_);
        return 1;
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
    for (int i = 0; i < self_N; ++i) {
        self_y->data[i] = y0_in->data[i];
    }

    self_k1 = oif_create_array_f64(1, y0_in->dimensions);
    self_k2 = oif_create_array_f64(1, y0_in->dimensions);
    self_k3 = oif_create_array_f64(1, y0_in->dimensions);
    self_k4 = oif_create_array_f64(1, y0_in->dimensions);
    self_k5 = oif_create_array_f64(1, y0_in->dimensions);
    self_k6 = oif_create_array_f64(1, y0_in->dimensions);

    self_y1 = oif_create_array_f64(1, y0_in->dimensions);
    self_ysti = oif_create_array_f64(1, y0_in->dimensions);

    self_sc = oif_create_array_f64(1, y0_in->dimensions);

    nfcn_ = 0;

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
    (void)integrator_name;
    (void)config_;
    fprintf(stderr, "[%s] Method `set_integrator` is not supported\n", prefix_);
    return 0;
}

int
print_stats(void)
{
    printf("[%s] Number of right-hand side evaluations = %zu\n", prefix_, nfcn_);
    return 0;
}

/**
 * Write into `y` the updated solution at time `t`.
 *
 */
int
integrate(double t_, OIFArrayF64 *y_out)
{
    if (t_ <= self_t) {
        fprintf(stderr, "[%s] Time should be larger than the current time\n", prefix_);
    }

    FACOLD = 1e-4;
    EXP01 = 0.2 - BETA * 0.75;
    FACC1 = 1.0 / FACMIN;
    FACC2 = 1.0 / FACMAX;

    size_t nstep = 0;

    bool last = false;
    // 1st stage
    OIF_RHS_FN(self_t, self_y, self_k1, self_user_data);

    // It is not clear why 2, but this is from the Hairer's code.
    nfcn_ += 2;

    while (self_t < t_) {
        if (self_t + self_h >= t_) {
            self_h = t_ - self_t;
            last = true;
        }
        nstep++;

        // 2nd stage
        for (int i = 0; i < self_N; ++i) {
            self_y1->data[i] = self_y->data[i] + a2[1] * self_h * self_k1->data[i];
        }
        OIF_RHS_FN(self_t + C2 * self_h, self_y1, self_k2, self_user_data);

        // 3rd stage
        for (int i = 0; i < self_N; ++i) {
            self_y1->data[i] = self_y->data[i] +
                               self_h * (a3[1] * self_k1->data[i] + a3[2] * self_k2->data[i]);
        }
        OIF_RHS_FN(self_t + C3 * self_h, self_y1, self_k3, self_user_data);

        // 4th stage
        for (int i = 0; i < self_N; ++i) {
            self_y1->data[i] = self_y->data[i] +
                               self_h * (a4[1] * self_k1->data[i] + a4[2] * self_k2->data[i] +
                                         a4[3] * self_k3->data[i]);
        }
        OIF_RHS_FN(self_t + C4 * self_h, self_y1, self_k4, self_user_data);

        // 5th stage
        for (int i = 0; i < self_N; ++i) {
            self_y1->data[i] = self_y->data[i] +
                               self_h * (a5[1] * self_k1->data[i] + a5[2] * self_k2->data[i] +
                                         a5[3] * self_k3->data[i] + a5[4] * self_k4->data[i]);
        }
        OIF_RHS_FN(self_t + C5 * self_h, self_y1, self_k5, self_user_data);

        // step 6.
        for (int i = 0; i < self_N; ++i) {
            self_ysti->data[i] =
                self_y->data[i] +
                self_h * (a6[1] * self_k1->data[i] + a6[2] * self_k2->data[i] +
                          a6[3] * self_k3->data[i] + a6[4] * self_k4->data[i] +
                          a6[5] * self_k5->data[i]);
        }
        OIF_RHS_FN(self_t + self_h, self_ysti, self_k6, self_user_data);

        // Estimate less accurate.
        for (int i = 0; i < self_N; ++i) {
            self_y1->data[i] = self_y->data[i] +
                               self_h * (a7[1] * self_k1->data[i] + a7[3] * self_k3->data[i] +
                                         a7[4] * self_k4->data[i] + a7[5] * self_k5->data[i] +
                                         a7[6] * self_k6->data[i]);
        }
        OIF_RHS_FN(self_t + self_h, self_y1, self_k2, self_user_data);

        nfcn_ += 6;
        // --------------------------------------------------------------------
        // Error estimation.
        double err = 0.0;

        for (int i = 0; i < self_N; ++i) {
            self_k4->data[i] = self_h * (e[1] * self_k1->data[i] + e[3] * self_k3->data[i] +
                                         e[4] * self_k4->data[i] + e[5] * self_k5->data[i] +
                                         e[6] * self_k6->data[i] + e[7] * self_k2->data[i]);
            double sk =
                self_atol + self_rtol * MAX(fabs(self_y->data[i]), fabs(self_y1->data[i]));
            err += pow(self_k4->data[i] / sk, 2);
        }
        err = sqrt(err / self_N);

        double FAC11 = pow(err, EXP01);
        // Lund stabilization.
        FAC = FAC11 / pow(FACOLD, BETA);
        // We require FACMIN <= HNEW/H <= FACMAX
        FAC = MAX(FACC2, MIN(FACC1, FAC / SAFE));
        double hnew = self_h / FAC;

        if (err < 1.0) {
            // Step is accepted.
            FACOLD = MAX(err, 1.0e-4);
            self_t += self_h;

            OIFArrayF64 *tmp = NULL;

            // Instead of copying `self_k2` into `self_k1` we just swap pointers.
            tmp = self_k1;
            self_k1 = self_k2;
            self_k2 = tmp;

            // Instead of copying `self_y1` into `self_y` we just swap pointers.
            tmp = self_y;
            self_y = self_y1;
            self_y1 = tmp;

            if (last) {
                for (int i = 0; i < self_N; ++i) {
                    y_out->data[i] = self_y->data[i];
                }
            }
            n_rejected = 0;
        }
        else {
            // Solution rejected.
            // It is also advisable to put facmax = 1 in the steps
            // right after a step-rejection (Shampine & Watts 1979).
            // Hairer et. al., vol. 1, p. 168.
            /* FACMAX = 1.0; */
            hnew = self_h / MIN(FACC1, FAC11 / SAFE);
            n_rejected++;
        }

        self_h = hnew;

        if (n_rejected > 20) {
            fprintf(stderr, "[%s::integrate] Too many rejected steps\n", prefix_);
            exit(1);
        }
    }

    return 0;
}

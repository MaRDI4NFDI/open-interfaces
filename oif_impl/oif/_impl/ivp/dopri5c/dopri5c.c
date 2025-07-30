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
#include <math.h>
#include <tgmath.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include <oif_impl/ivp.h>
#include <oif/c_bindings.h>
#include <oif/util.h>

#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

static char const *const prefix_ = "ivp::dopri5c";

typedef struct self {
    int N;            // Length of the solution vector.
    double t;
    double h;      // Current step size.
    double rtol;   // relative tolerance
    double atol;  // absolute tolerance

    // Runge--Kutta stages.
    OIFArrayF64 *k1;
    OIFArrayF64 *k2;
    OIFArrayF64 *k3;
    OIFArrayF64 *k4;
    OIFArrayF64 *k5;
    OIFArrayF64 *k6;

    // Solution vectors to avoid copying.
    OIFArrayF64 *y;
    OIFArrayF64 *y1;
    OIFArrayF64 *ysti;

    OIFArrayF64 *sc;

    // User data that are passed to the right-hand side function.
    void *user_data;

    rhs_fn_t rhs_fn;

    // Number of right-hand side function evaluations.
    size_t nfcn_;
    // Number of rejected steps.
    size_t n_rejected;

    double FACOLD;
} Self;

// Signature for the right-hand side that is provided by the `IVP` interface.


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

static void
compute_initial_step_(Self *self)
{
    double h0;
    double h1;

    self->rhs_fn(self->t, self->y, self->k1, self->user_data);

    for (int i = 0; i < self->N; ++i) {
        self->sc->data[i] = self->atol + self->y->data[i] * self->rtol;
    }

    double d0 = 0.0L;
    for (int i = 0; i < self->N; ++i) {
        d0 += pow(self->y->data[i] / self->sc->data[i], 2);
    }
    d0 = sqrt(d0 / self->N);

    double d1 = 0.0L;
    for (int i = 0; i < self->N; ++i) {
        d1 += pow(self->k1->data[i] / self->sc->data[i], 2);
    }
    d1 = sqrt(d1 / self->N);

    if (d0 < 1e-5 || d1 < 1e-5) {
        h0 = 1e-6;
    }
    else {
        h0 = 1e-2 * (d0 / d1);
    }

    for (int i = 0; i < self->N; ++i) {
        self->y1->data[i] = self->y->data[i] + h0 * self->k1->data[i];
    }
    self->rhs_fn(self->t + self->h, self->y1, self->k2, self->user_data);

    double d2 = 0.0L;
    for (int i = 0; i < self->N; ++i) {
        d2 += pow((self->k2->data[i] - self->k1->data[i]) / self->sc->data[i], 2.0);
    }
    d2 = sqrt(d2 / self->N) / h0;

    if (MAX(d1, d2) < 1e-15) {
        h1 = MAX(1e-6, h0 * 1e-3);
    }
    else {
        h1 = pow(1e-2 / MAX(d1, d2), 1.0L / 5.0L);
    }

    self->h = MIN(1e2 * h0, h1);
}

Self *
malloc_self(void)
{
    Self *self = oif_util_malloc(sizeof(Self));
    if (self == NULL) {
        fprintf(stderr, "[%s] Could not allocate memory for Self object\n", prefix_);
        return NULL;
    }

    self->N = 0;
    self->t = 0.0;
    self->h = 0.0;
    self->rtol = 1e-6;
    self->atol = 1e-12;

    self->k1 = NULL;
    self->k2 = NULL;
    self->k3 = NULL;
    self->k4 = NULL;
    self->k5 = NULL;
    self->k6 = NULL;

    self->y = NULL;
    self->y1 = NULL;
    self->ysti = NULL;

    self->sc = NULL;

    self->user_data = NULL;
    self->rhs_fn = NULL;

    self->nfcn_ = 0;
    self->n_rejected = 0;

    return self;
}

int
set_initial_value(Self *self, OIFArrayF64 *y0_in, double t0_in)
{
    (void)self;
    EXP01 = 0.2L - BETA * 0.75L;
    FACC1 = 1.0L / FACMIN;
    self->t = t0_in;

    if (y0_in->nd != 1) {
        fprintf(stderr, "[%s] Accept only one-dimensional arrays (vectors)\n", prefix_);
        return 1;
    }

    if (self->y) {
        oif_free_array_f64(self->y);
        oif_free_array_f64(self->k1);
        oif_free_array_f64(self->k2);
        oif_free_array_f64(self->k3);
        oif_free_array_f64(self->k4);
        oif_free_array_f64(self->k5);
        oif_free_array_f64(self->k6);

        oif_free_array_f64(self->y1);
        oif_free_array_f64(self->ysti);

        oif_free_array_f64(self->sc);
    }

    self->N = y0_in->dimensions[0];

    self->y = oif_create_array_f64(1, y0_in->dimensions);
    for (int i = 0; i < self->N; ++i) {
        self->y->data[i] = y0_in->data[i];
    }

    self->k1 = oif_create_array_f64(1, y0_in->dimensions);
    self->k2 = oif_create_array_f64(1, y0_in->dimensions);
    self->k3 = oif_create_array_f64(1, y0_in->dimensions);
    self->k4 = oif_create_array_f64(1, y0_in->dimensions);
    self->k5 = oif_create_array_f64(1, y0_in->dimensions);
    self->k6 = oif_create_array_f64(1, y0_in->dimensions);

    self->y1 = oif_create_array_f64(1, y0_in->dimensions);
    self->ysti = oif_create_array_f64(1, y0_in->dimensions);

    self->sc = oif_create_array_f64(1, y0_in->dimensions);

    self->nfcn_ = 0;

    return 0;
}

int
set_user_data(Self *self, void *user_data)
{
    // We did not take an ownership of `user_data`,
    // so we do not free it here, even if it was already set.
    // It is the responsibility of the user to free it.
    self->user_data = user_data;
    return 0;
}

int
set_rhs_fn(Self *self, rhs_fn_t rhs)
{
    if (rhs == NULL) {
        fprintf(stderr, "[%s] `set_rhs_fn` accepts non-null function pointer only\n", prefix_);
        return 1;
    }
    self->rhs_fn = rhs;
    compute_initial_step_(self);
    return 0;
}

int
set_tolerances(Self *self, double rtol, double atol)
{
    (void)self;  // Unused in this implementation.
    self->rtol = rtol;
    self->atol = atol;
    return 0;
}

int
set_integrator(Self *self, const char *integrator_name, OIFConfigDict *config_)
{
    (void)self;  // Unused in this implementation.
    (void)integrator_name;
    (void)config_;
    fprintf(stderr, "[%s] Method `set_integrator` is not supported\n", prefix_);
    return 1;
}

int
print_stats(Self *self)
{
    (void)self;  // Unused in this implementation.
    printf("[%s] Number of right-hand side evaluations = %zu\n", prefix_, self->nfcn_);
    return 0;
}

/**
 * Write into `y_out` the updated solution at time `t`.
 *
 */
int
integrate(Self *self, double t_, OIFArrayF64 *y_out)
{
    (void)self;  // Unused in this implementation.
    if (t_ <= self->t) {
        fprintf(stderr, "[%s] Time should be larger than the current time\n", prefix_);
    }

    FACOLD = 1e-4;
    EXP01 = 0.2 - BETA * 0.75;
    FACC1 = 1.0 / FACMIN;
    FACC2 = 1.0 / FACMAX;

    size_t nstep = 0;

    bool last = false;
    // 1st stage
    self->rhs_fn(self->t, self->y, self->k1, self->user_data);

    // It is not clear why 2, but this is from the Hairer's code.
    self->nfcn_ += 2;

    while (self->t < t_) {
        if (self->t + self->h >= t_) {
            self->h = t_ - self->t;
            last = true;
        }
        nstep++;

        // 2nd stage
        for (int i = 0; i < self->N; ++i) {
            self->y1->data[i] = self->y->data[i] + a2[1] * self->h * self->k1->data[i];
        }
        self->rhs_fn(self->t + C2 * self->h, self->y1, self->k2, self->user_data);

        // 3rd stage
        for (int i = 0; i < self->N; ++i) {
            self->y1->data[i] = self->y->data[i] +
                               self->h * (a3[1] * self->k1->data[i] + a3[2] * self->k2->data[i]);
        }
        self->rhs_fn(self->t + C3 * self->h, self->y1, self->k3, self->user_data);

        // 4th stage
        for (int i = 0; i < self->N; ++i) {
            self->y1->data[i] = self->y->data[i] +
                               self->h * (a4[1] * self->k1->data[i] + a4[2] * self->k2->data[i] +
                                         a4[3] * self->k3->data[i]);
        }
        self->rhs_fn(self->t + C4 * self->h, self->y1, self->k4, self->user_data);

        // 5th stage
        for (int i = 0; i < self->N; ++i) {
            self->y1->data[i] = self->y->data[i] +
                               self->h * (a5[1] * self->k1->data[i] + a5[2] * self->k2->data[i] +
                                         a5[3] * self->k3->data[i] + a5[4] * self->k4->data[i]);
        }
        self->rhs_fn(self->t + C5 * self->h, self->y1, self->k5, self->user_data);

        // step 6.
        for (int i = 0; i < self->N; ++i) {
            self->ysti->data[i] =
                self->y->data[i] +
                self->h * (a6[1] * self->k1->data[i] + a6[2] * self->k2->data[i] +
                          a6[3] * self->k3->data[i] + a6[4] * self->k4->data[i] +
                          a6[5] * self->k5->data[i]);
        }
        self->rhs_fn(self->t + self->h, self->ysti, self->k6, self->user_data);

        // Estimate less accurate.
        for (int i = 0; i < self->N; ++i) {
            self->y1->data[i] = self->y->data[i] +
                               self->h * (a7[1] * self->k1->data[i] + a7[3] * self->k3->data[i] +
                                         a7[4] * self->k4->data[i] + a7[5] * self->k5->data[i] +
                                         a7[6] * self->k6->data[i]);
        }
        self->rhs_fn(self->t + self->h, self->y1, self->k2, self->user_data);

        self->nfcn_ += 6;
        // --------------------------------------------------------------------
        // Error estimation.
        double err = 0.0;

        for (int i = 0; i < self->N; ++i) {
            self->k4->data[i] = self->h * (e[1] * self->k1->data[i] + e[3] * self->k3->data[i] +
                                         e[4] * self->k4->data[i] + e[5] * self->k5->data[i] +
                                         e[6] * self->k6->data[i] + e[7] * self->k2->data[i]);
            double sk =
                self->atol + self->rtol * MAX(fabs(self->y->data[i]), fabs(self->y1->data[i]));
            err += pow(self->k4->data[i] / sk, 2);
        }
        err = sqrt(err / self->N);

        double FAC11 = pow(err, EXP01);
        // Lund stabilization.
        FAC = FAC11 / pow(FACOLD, BETA);
        // We require FACMIN <= HNEW/H <= FACMAX
        FAC = MAX(FACC2, MIN(FACC1, FAC / SAFE));
        double hnew = self->h / FAC;

        if (err < 1.0) {
            // Step is accepted.
            FACOLD = MAX(err, 1.0e-4);
            self->t += self->h;

            OIFArrayF64 *tmp = NULL;

            // Instead of copying `self_k2` into `self_k1` we just swap pointers.
            tmp = self->k1;
            self->k1 = self->k2;
            self->k2 = tmp;

            // Instead of copying `self_y1` into `self_y` we just swap pointers.
            tmp = self->y;
            self->y = self->y1;
            self->y1 = tmp;

            if (last) {
                for (int i = 0; i < self->N; ++i) {
                    y_out->data[i] = self->y->data[i];
                }
            }
            self->n_rejected = 0;
        }
        else {
            // Solution rejected.
            // It is also advisable to put facmax = 1 in the steps
            // right after a step-rejection (Shampine & Watts 1979).
            // Hairer et. al., vol. 1, p. 168.
            /* FACMAX = 1.0; */
            hnew = self->h / MIN(FACC1, FAC11 / SAFE);
            self->n_rejected++;
        }

        self->h = hnew;

        if (self->n_rejected > 20) {
            fprintf(stderr, "[%s::integrate] Too many rejected steps\n", prefix_);
            exit(1);
        }
    }

    return 0;
}

void
free_self(Self *self)
{
    if (self == NULL) {
        fprintf(stderr, "[%s] \033[31mERROR\033[0m Self is NULL, nothing to free\n", prefix_);
        return;
    }

    fprintf(stderr, "[%s] Freeing resources...\n", prefix_);
    fprintf(stderr, "[%s] self->k1 = %p\n", prefix_, self->k1);
    if (self->k1 != NULL) {
        oif_free_array_f64(self->k1);
    }
    if (self->k2 != NULL) {
        oif_free_array_f64(self->k2);
    }
    if (self->k3 != NULL) {
        oif_free_array_f64(self->k3);
    }
    if (self->k4 != NULL) {
        oif_free_array_f64(self->k4);
    }
    if (self->k5 != NULL) {
        oif_free_array_f64(self->k5);
    }
    if (self->k6 != NULL) {
        oif_free_array_f64(self->k6);
    }

    if (self->y != NULL) {
        oif_free_array_f64(self->y);
    }
    if (self->y1 != NULL) {
        oif_free_array_f64(self->y1);
    }
    if (self->ysti != NULL) {
        oif_free_array_f64(self->ysti);
    }

    if (self->sc != NULL) {
        oif_free_array_f64(self->sc);
    }

    oif_util_free(self);
    fprintf(stderr, "[%s] Freeing resources...DONE!\n", prefix_);
}

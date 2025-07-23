/**
 * Implementation of the `ivp` interface with Sundials CVODE solver.
 * CVODE solver is an advanced solver that can solve nonstiff problems
 * using Adams multistep method and stiff problems using BDF method.
 * See https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html
 *
 * This code uses the following types from Sundials:
 * - sunrealtype – the floating-point type
 * - sunindextype – the integer type used for vector and matrix indices
 */
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include "oif/api.h"
#include "oif/config_dict.h"
#include "oif/util.h"
#include "oif_impl/ivp.h"

const char *prefix = "[ivp::sundials_cvode]";

// Signature for the right-hand side function that CVode expects.
static int
cvode_rhs(sunrealtype t, N_Vector u, N_Vector u_dot, void *user_data);

/*
 * In Sundials 7.0, `SUNContext_Create` accepts `SUNComm` instead of `void *`
 * as the first argument.
 * This is required for compatibility with versions <7.0.
 */
#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif

static const char *AVAILABLE_OPTIONS_[] = {
    "max_num_steps",
    NULL,
};

typedef struct self {
    /** Number of equations */
    sunindextype N;
    sunrealtype t0;
    N_Vector y0;
    sunrealtype rtol;  // relative tolerance
    sunrealtype atol;  // absolute tolerance
    int integrator;
    // Signature for the right-hand side that is provided by the `IVP` interface
    // of the Open Interfaces.
    rhs_fn_t OIF_RHS_FN;
    OIFConfigDict *config;
    void *user_data;
    // Global state of the module.
    // Sundials context
    SUNContext sunctx;
    // CVode memory block.
    void *cvode_mem;
    SUNNonlinearSolver NLS;
} Self;

Self *
malloc_self(void)
{
    Self *self = (Self *)malloc(sizeof(Self));
    if (self == NULL) {
        fprintf(stderr, "%s Could not allocate memory for Self object\n", prefix);
        return NULL;
    }
    self->N = 0;
    self->t0 = 0.0;
    self->y0 = NULL;
    self->rtol = 1e-6;
    self->atol = 1e-12;
    self->integrator = CV_ADAMS;
    self->OIF_RHS_FN = NULL;
    self->config = NULL;
    self->user_data = NULL;
    self->sunctx = NULL;
    self->cvode_mem = NULL;
    self->NLS = NULL;

    return self;
}

int
set_user_data(Self *self, void *user_data);

static int
init_(Self *self)
{
    int status;

    // Clean up existing resources before reinitializing
    if (self->cvode_mem != NULL) {
        CVodeFree(&self->cvode_mem);
        self->cvode_mem = NULL;
    }

    // 5. Create CVODE object.
    self->cvode_mem = CVodeCreate(self->integrator, self->sunctx);
    if (self->cvode_mem == NULL) {
        fprintf(stderr, "%s CVodeCreate failed\n", prefix);
        return 1;
    }

    // 6. Initialize CVODE solver.
    status = CVodeInit(self->cvode_mem, cvode_rhs, self->t0, self->y0);
    if (status) {
        fprintf(stderr, "%s CVodeInit call failed\n", prefix);
        return 1;
    }

    // 7. Specify integration tolerances.
    status = CVodeSStolerances(self->cvode_mem, self->rtol, self->atol);
    if (status) {
        fprintf(stderr, "%s CVodeSStolerances failed\n", prefix);
        return 1;
    }

    // 8. Create matrix object
    /* A = SUNDenseMatrix(N, N, sunctx); */
    /* if (A == NULL) { */
    /*     fprintf(stderr, "[sundials_cvode] Could not create matrix for dense linear
     * solver\n"); */
    /*     return 2; */
    /* } */

    // 9. Create linear solver object
    /* SUNLinearSolver linear_solver = */
    /*     SUNLinSol_Dense(y0, A, sunctx); */
    /* if (linear_solver == NULL) { */
    /*     fprintf(stderr, */
    /*             "%s An error occurred when creating SUNLinearSolver", */
    /*             prefix); */
    /*     return 3; */
    /* } */

    // 10. Set linear solver optional inputs

    // 11. Attach linear solver module.
    /* status = CVodeSetLinearSolver(cvode_mem, linear_solver, A); */
    /* if (status == CVLS_MEM_FAIL) { */
    /*     fprintf( */
    /*         stderr, */
    /*         "[sundials_cvode] Setting linear solver failed\n" */
    /*     ); */
    /*     return 4; */
    /* } else if (status == CVLS_ILL_INPUT) { */
    /*     fprintf( */
    /*         stderr, */
    /*         "[sundials_cvode] Setting linear solver failed due to ill input\n" */
    /*     ); */
    /*     return 5; */
    /* } else if (status != CVLS_SUCCESS) { */
    /*     fprintf( */
    /*         stderr, */
    /*         "[sundials_cvode] Setting linear solver was unsuccessful\n" */
    /*     ); */
    /*     return 6; */
    /* } */

    // 12. Set optional inputs
    if (self->config != NULL) {
        if (oif_config_dict_key_exists(self->config, "max_num_steps")) {
            int max_num_steps = oif_config_dict_get_int(self->config, "max_num_steps");
            status = CVodeSetMaxNumSteps(self->cvode_mem, max_num_steps);
            if (status) {
                fprintf(stderr, "%s Could not set max number of steps\n", prefix);
                return 1;
            }
        }
    }

    if (self->user_data != NULL) {
        status = set_user_data(self, self->user_data);
        if (status) {
            return 1;
        }
    }

    // 13. Create nonlinear solver object (optional)
    if (self->NLS != NULL) {
        SUNNonlinSolFree(self->NLS);
        self->NLS = NULL;
    }
    self->NLS = SUNNonlinSol_FixedPoint(self->y0, 0, self->sunctx);
    if (self->NLS == NULL) {
        fprintf(stderr, "%s Could not create Fixed Point Nonlinear solver\n", prefix);
        return 7;
    }

    // 14. Attach nonlinear solver module (optional)
    status = CVodeSetNonlinearSolver(self->cvode_mem, self->NLS);
    if (status != CV_SUCCESS) {
        fprintf(stderr, "%s CVodeSetNonlinearSolver failed with code %d\n", prefix, status);
        return 8;
    }
    // 15. Set nonlinear solver optional inputs (optional)
    // 16. Specify rootfinding problem (optional)

    return 0;
}

int
set_initial_value(Self *self, OIFArrayF64 *y0_in, double t0_in)
{
    if ((y0_in == NULL) || (y0_in->data == NULL)) {
        fprintf(stderr, "`set_initial_value` received NULL argument\n");
        exit(1);
    }
    int status;  // Check errors

    // 1. Initialize parallel or multi-threaded environment, if appropriate.
    // No, it is not appropriate here as we work with serial code :-)

    // 2. Create the Sundials context object.
    if (self->sunctx != NULL) {
        SUNContext_Free(&self->sunctx);
    }
    status = SUNContext_Create(SUN_COMM_NULL, &self->sunctx);
    if (status) {
        fprintf(stderr, "%s An error occurred when creating SUNContext\n", prefix);
        return 1;
    }

    // 3. Set problem dimensions, etc.
    if (sizeof(SUNDIALS_INDEX_TYPE) == sizeof(int)) {
        if (y0_in->dimensions[0] > INT_MAX) {
            fprintf(stderr,
                    "[sundials_cvode] Dimensions of the array are larger "
                    "than the internal Sundials type 'sunindextype'\n");
            return 1;
        }
        else {
            self->N = (sunindextype)y0_in->dimensions[0];
        }
    }
    else {
        fprintf(stderr,
                "[sundials_cvode] Assumption that the internal Sundials type "
                "'sundindextype' is 'int' is violated. Cannot proceed\n");
        return 2;
    }

    // 4. Set vector of initial values.
    if (self->y0 != NULL) {
        N_VDestroy(self->y0);
    }

    self->y0 = N_VMake_Serial(self->N, y0_in->data, self->sunctx);  // Problem vector.
    if (self->y0 == NULL) {
        fprintf(stderr, "%s Could not create initial value vector\n", prefix);
        return 3;
    }

    // Sanity check that `sunrealtype` is actually the same as OIF_FLOAT64.
    assert(NV_Ith_S(self->y0, 0) == y0_in->data[0]);

    self->t0 = t0_in;
    assert(self->t0 == t0_in);

    status = init_(self);
    if (status) {
        return status;
    }

    return 0;
}

int
set_user_data(Self *self, void *user_data)
{
    self->user_data = user_data;
    int status = CVodeSetUserData(self->cvode_mem, self);
    if (status == CV_MEM_NULL) {
        fprintf(stderr,
                "%s Could not set user data as "
                "CVODE memory block is not yet initialized\n",
                prefix);
        return 1;
    }
    assert(status == CV_SUCCESS);
    return 0;
}

int
set_rhs_fn(Self *self, rhs_fn_t rhs)
{
    if (rhs == NULL) {
        fprintf(stderr, "`set_rhs_fn` accepts non-null function pointer only\n");
        return 1;
    }
    self->OIF_RHS_FN = rhs;
    return 0;
}

int
set_tolerances(Self *self, double rtol, double atol)
{
    self->rtol = rtol;
    self->atol = atol;
    CVodeSStolerances(self->cvode_mem, self->rtol, self->atol);
    return 0;
}

int
set_integrator(Self *self, const char *integrator_name, OIFConfigDict *config_)
{
    if (oif_strcmp_nocase(integrator_name, "bdf") == 0) {
        self->integrator = CV_BDF;
    }
    else if (oif_strcmp_nocase(integrator_name, "adams") == 0) {
        self->integrator = CV_ADAMS;
    }
    else {
        fprintf(stderr, "[%s] Supported values for integrator name are `bdf` and `adams`\n",
                prefix);
        return 1;
    }

    if (config_ != NULL) {
        const char **keys = oif_config_dict_get_keys(config_);
        for (int i = 0; keys[i] != NULL; i++) {
            bool found = false;
            for (int j = 0; AVAILABLE_OPTIONS_[j] != NULL; j++) {
                if (strcmp(keys[i], AVAILABLE_OPTIONS_[j]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr,
                        "[%s] Passed option '%s' is not one of the available options\n",
                        prefix, keys[i]);
                oif_util_free(keys);
                oif_config_dict_free(config_);
                return 1;
            }
        }
        oif_util_free(keys);
    }

    // Clean up existing config if it exists
    if (self->config != NULL) {
        oif_config_dict_free(self->config);
    }
    self->config = config_;

    if (self->y0 != NULL) {
        int status = init_(self);
        if (status) {
            return status;
        }
    }

    return 0;
}

int
print_stats(Self *self)
{
    if (self->cvode_mem == NULL) {
        fprintf(stderr, "%s CVODE not initialized\n", prefix);
        return 1;
    }
    return CVodePrintAllStats(self->cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
}

int
integrate(Self *self, double t, OIFArrayF64 *y)
{
    /* if ((y == NULL) || (y->data == NULL)) { */
    /*     fprintf(stderr, "`integrate` received NULL argument\n"); */
    /*     exit(1); */
    /* } */
    int ier;  // Error checking.

    N_Vector yout = N_VMake_Serial(self->N, y->data, self->sunctx);
    sunrealtype tout = t;

    // Time that will be reached by solver during integration.
    // When we request CV_NORMAL task, it must be close to requested time
    // `tout`.
    // When we request CV_ONE_STEP task, than it will be just time reached
    // via internal time step (time step that satisfies error tolerances).
    sunrealtype tret;

    // 17. Advance solution in time.
    ier = CVode(self->cvode_mem, tout, yout, &tret, CV_NORMAL);
    N_VDestroy(yout);
    // TODO: Handle all cases: write good error messages for all `ier`.
    switch (ier) {
        case CV_SUCCESS:
            return 0;
        default:
            fprintf(stderr, "%s During call to `CVode`, an error occurred (code: %d)\n",
                    prefix, ier);
            return 1;
    }
}

// Function that computes the right-hand side of the ODE system.
static int
cvode_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *context)
{
    // Because we use object-oriented approach,
    // we need to abuse a little bit Sundials' assumptions
    // about what `user_data` mean (the last argument).
    // We actually pass a pointer to our `Self` object
    // as user data, and the `Self` object contains
    // pointers to the user-provided right-hand side function
    // and its user data.
    Self *self = (Self *)context;

    // While Sundials CVode works with `N_Vector` data structure
    // for one-dimensional arrays, the user provides right-hand side
    // function that works with `OIFArrayF64` data structure,
    // so we need to convert between them here.

    // Construct OIFArrayF64 to pass to the user-provided right-hand side
    // function.
    OIFArrayF64 oif_y = {
        .nd = 1, .dimensions = (intptr_t[]){N_VGetLength(y)}, .data = N_VGetArrayPointer(y)};
    OIFArrayF64 oif_ydot = {.nd = 1,
                            .dimensions = (intptr_t[]){N_VGetLength(ydot)},
                            .data = N_VGetArrayPointer(ydot)};

    int result = self->OIF_RHS_FN(t, &oif_y, &oif_ydot, self->user_data);

    return result;
}

int
oif_ivp_free(void *self_)
{
    fprintf(stderr, "%s Freeing resources\n", prefix);
    Self *self = (Self *)self_;  // Fixed: cast to void to suppress unused parameter warning
    assert(self != NULL);

    if (self->y0 != NULL) {
        N_VDestroy(self->y0);
        self->y0 = NULL;
    }

    if (self->cvode_mem != NULL) {
        CVodeFree(&self->cvode_mem);
        self->cvode_mem = NULL;
    }

    if (self->NLS != NULL) {
        SUNNonlinSolFree(self->NLS);
        self->NLS = NULL;
    }

    if (self->sunctx != NULL) {
        SUNContext_Free(&self->sunctx);
        self->sunctx = NULL;
    }

    if (self->config != NULL) {
        oif_config_dict_free(self->config);
        self->config = NULL;
    }

    free(self);
    self = NULL;

    return 0;
}

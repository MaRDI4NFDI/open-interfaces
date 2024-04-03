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

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include "oif/api.h"
#include "oif_impl/ivp.h"

const char *prefix = "[ivp::sundials_cvode]";

// Signature for the right-hand side that is provided by the `IVP` interface
// of the Open Interfaces.
static oif_ivp_rhs_fn_t OIF_RHS_FN;

// Signature for the right-hand side function that CVode expects.
static int
cvode_rhs(sunrealtype t, N_Vector u, N_Vector u_dot, void *user_data);

// Global state of the module.
// Sundials context
static SUNContext sunctx;
// CVode memory block.
void *cvode_mem;

/** Number of equations */
sunindextype N;

/*
 * In Sundials 7.0, `SUNContext_Create` accepts `SUNComm` instead of `void *`
 * as the first argument.
 * This is required for compatibility with versions <7.0.
 */
#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif

int
set_initial_value(OIFArrayF64 *y0_in, double t0_in)
{
    if ((y0_in == NULL) || (y0_in->data == NULL)) {
        fprintf(stderr, "`set_initial_value` received NULL argument\n");
        exit(1);
    }
    int status;                  // Check errors
    sunrealtype abstol = 1e-15;  // absolute tolerance
    sunrealtype reltol = 1e-15;  // relative tolerance

    // 1. Initialize parallel or multi-threaded environment, if appropriate.
    // No, it is not appropriate here as we work with serial code :-)

    // 2. Create the Sundials context object.
    status = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    if (status) {
        fprintf(stderr, "%s An error occurred when creating SUNContext", prefix);
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
            N = (sunindextype)y0_in->dimensions[0];
        }
    }
    else {
        fprintf(stderr,
                "[sundials_cvode] Assumption that the internal Sundials type "
                "'sundindextype' is 'int' is violated. Cannot proceed\n");
        return 2;
    }

    // 4. Set vector of initial values.
    N_Vector y0 = N_VMake_Serial(N, y0_in->data, sunctx);  // Problem vector.
    // Sanity check that `sunrealtype` is actually the same as OIF_FLOAT64.
    assert(NV_Ith_S(y0, 0) == y0_in->data[0]);

    sunrealtype t0 = t0_in;
    assert(t0 == t0_in);

    // 5. Create CVODE object.
    cvode_mem = CVodeCreate(CV_ADAMS, sunctx);

    // 6. Initialize CVODE solver.
    status = CVodeInit(cvode_mem, cvode_rhs, t0, y0);
    if (status) {
        fprintf(stderr, "%s CVodeInit call failed", prefix);
        return 1;
    }

    // 7. Specify integration tolerances.
    CVodeSStolerances(cvode_mem, reltol, abstol);

    // 8. Create matrix object
    /* A = SUNDenseMatrix(N, N, sunctx); */
    /* if (A == NULL) { */
    /*     fprintf(stderr, "[sundials_cvode] Could not create matrix for dense
     * linear solver\n"); */
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
    /*         "[sundials_cvode] Setting linear solver failed due to ill
     * input\n" */
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
    // 13. Create nonlinear solver object (optional)
    SUNNonlinearSolver NLS = SUNNonlinSol_FixedPoint(y0, 0, sunctx);
    if (NLS == NULL) {
        fprintf(stderr, "%s Could not create Fixed Point Nonlinear solver\n", prefix);
        return 7;
    }
    // 14. Attach nonlinear solver module (optional)
    status = CVodeSetNonlinearSolver(cvode_mem, NLS);
    if (status != CV_SUCCESS) {
        fprintf(stderr, "%s CVodeSetNonlinearSolver failed with code %d\n", prefix, status);
        return 8;
    }
    // 15. Set nonlinear solver optional inputs (optional)
    // 16. Specify rootfinding problem (optional)

    N_VDestroy_Serial(y0);

    return 0;
}

int
set_user_data(void *user_data)
{
    int status = CVodeSetUserData(cvode_mem, user_data);
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
set_rhs_fn(oif_ivp_rhs_fn_t rhs)
{
    if (rhs == NULL) {
        fprintf(stderr, "`set_rhs_fn` accepts non-null function pointer only\n");
        return 1;
    }
    OIF_RHS_FN = rhs;
    return 0;
}

int
set_tolerances(double rtol, double atol)
{
    CVodeSStolerances(cvode_mem, rtol, atol);
    return 0;
}

int
print_stats(void)
{
    return CVodePrintAllStats(cvode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
}

int
integrate(double t, OIFArrayF64 *y)
{
    /* if ((y == NULL) || (y->data == NULL)) { */
    /*     fprintf(stderr, "`integrate` received NULL argument\n"); */
    /*     exit(1); */
    /* } */
    int ier;  // Error checking.

    N_Vector yout = N_VMake_Serial(N, y->data, sunctx);
    sunrealtype tout = t;

    // Time that will be reached by solver during integration.
    // When we request CV_NORMAL task, it must be close to requested time
    // `tout`.
    // When we request CV_ONE_STEP task, than it will be just time reached
    // via internal time step (time step that satisfies error tolerances).
    sunrealtype tret;

    // 17. Advance solution in time.
    ier = CVode(cvode_mem, tout, yout, &tret, CV_NORMAL);
    N_VDestroy(yout);
    // TODO: Handle all cases: write good error messages for all `ier`.
    switch (ier) {
        case CV_SUCCESS:
            return 0;
        default:
            fprintf(stderr, "%s During call to `CVode`, an error occurred\n", prefix);
            return 1;
    }
}

// Function that computes the right-hand side of the ODE system.
static int
cvode_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
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

    int result = OIF_RHS_FN(t, &oif_y, &oif_ydot, user_data);

    return result;
}

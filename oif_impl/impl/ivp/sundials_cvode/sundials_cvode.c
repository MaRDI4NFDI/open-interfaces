/**
 * Implementation of the `ivp` interface with Sundials CVODE solver.
 * CVODE solver is an advanced solver that can solve nonstiff problems
 * using Adams multistep method and stiff problems using BDF method.
 * See https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html
 * Big thank you to the people that created GitHub Copilot.
 *
 * This code uses the following types from Sundials:
 * - realtype – the floating-point type
 * - sunindextype – the integer type used for vector and matrix indices
 */
#include <assert.h>

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_spgmr.h>

#include "oif/api.h"

const char *prefix = "[impl::sundials_cvode]";

// Signature for the right-hand side that is provided by the `IVP` interface
// of the OpenInterFaces.
// TODO: Should be part of the interface header?
typedef void (*oif_rhs_fn_t)(double t, OIFArrayF64 *y, OIFArrayF64 *ydot);
static oif_rhs_fn_t OIF_RHS_FN;

// Signature for the right-hand side function that CVode expects.
static int cvode_rhs(realtype t, N_Vector u, N_Vector u_dot, void *user_data);

int set_rhs_fn(void rhs(double, OIFArrayF64 *y, OIFArrayF64 *y_dot)) {
    OIF_RHS_FN = rhs;
    return 0;
}

// Global state of the module.
// Sundials context
static SUNContext sunctx;
// CVode memory block.
void *cvode_mem;

int set_initial_value(OIFArrayF64 *y0_in, double t0_in) {
    if ((y0_in == NULL) || (y0_in->data == NULL)) {
        fprintf(stderr, "`set_initial_value` received NULL argument\n");
        exit(1);
    }
    int status;              // Check errors
    realtype abstol = 1e-15; // absolute tolerance
    realtype reltol = 1e-15; // relative tolerance

    // 1. Initialize parallel or multi-threaded environment, if appropriate.
    // No, it is not appropriate here as we work with serial code :-)

    // 2. Create the Sundials context object.
    status = SUNContext_Create(NULL, &sunctx);
    if (status) {
        fprintf(
            stderr, "%s An error occurred when creating SUNContext", prefix);
        return 1;
    }

    // 3. Set problem dimensions, etc.
    sunindextype N = y0_in->dimensions[0];

    // 4. Set vector of initial values.
    N_Vector y0 = N_VMake_Serial(N, y0_in->data, sunctx); // Problem vector.
    // Sanity check that `realtype` is actually the same as OIF_FLOAT64.
    assert(NV_Ith_S(y0, 0) == y0_in->data[0]);

    realtype t0 = t0_in;
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
    // pass

    // 9. Create linear solver object
    SUNLinearSolver linear_solver =
        SUNLinSol_SPGMR(y0, SUN_PREC_NONE, 0, sunctx);
    if (linear_solver == NULL) {
        fprintf(stderr,
                "%s An error occurred when creating SUNLinearSolver",
                prefix);
        return 1;
    }

    // 10. Set linear solver optional inputs

    // 11. Attach linear solver module.
    // NULL is because SPGMR is a matrix-free method, so no matrix is needed.
    CVodeSetLinearSolver(cvode_mem, linear_solver, NULL);

    // 12. Set optional inputs
    // 13. Create nonlinear solver object (optional)
    // 14. Attach nonlinear solver module (optional)
    // 15. Set nonlinear solver optional inputs (optional)
    // 16. Specify rootfinding problem (optional)

    N_VDestroy_Serial(y0);

    return 0;
}

int integrate(double t, OIFArrayF64 *y) {
    if ((y == NULL) || (y->data == NULL)) {
        fprintf(stderr, "`integrate` received NULL argument\n");
        exit(1);
    }
    int ier; // Error checking.

    sunindextype N = y->dimensions[0];

    N_Vector yout = N_VMake_Serial(N, y->data, sunctx);
    realtype tout = t;
    assert(tout == t);

    // Time that will be reached by solver during integration.
    // When we request CV_NORMAL task, it must be close to requested time
    // `tout`.
    // When we request CV_ONE_STEP task, than it will be just time reached
    // via internal time step (time step that satisfies error tolerances).
    realtype tret;

    // 17. Advance solution in time.
    ier = CVode(cvode_mem, tout, yout, &tret, CV_NORMAL);
    N_VDestroy(yout);
    // TODO: Handle all cases: write good error messages for all `ier`.
    switch (ier) {
    case CV_SUCCESS:
        return 0;
    default:
        fprintf(
            stderr, "%s During call to `CVode`, an error occurred\n", prefix);
        return 1;
    }
}

// Function that computes the right-hand side of the ODE system.
static int cvode_rhs(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
    // While Sundials CVode works with `N_Vector` data structure
    // for one-dimensional arrays, the user provides right-hand side
    // function that works with `OIFArrayF64` data structure,
    // so we need to convert between them here.

    // Construct OIFArrayF64 to pass to the user-provided right-hand side
    // function.
    OIFArrayF64 oif_y = {.nd = 1,
                         .dimensions = (intptr_t[]){N_VGetLength(y)},
                         .data = N_VGetArrayPointer(y)};
    OIFArrayF64 oif_ydot = {.nd = 1,
                            .dimensions = (intptr_t[]){N_VGetLength(ydot)},
                            .data = N_VGetArrayPointer(ydot)};

    OIF_RHS_FN(t, &oif_y, &oif_ydot);

    return 0;
}

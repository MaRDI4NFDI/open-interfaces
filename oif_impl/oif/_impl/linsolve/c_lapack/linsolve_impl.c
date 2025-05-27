#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Use LAPACKE - C-friendly interface to LAPACK.
#include <lapacke.h>

#include <oif/api.h>
#include <oif_impl/linsolve.h>

int
solve_lin(OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x)
{
    lapack_int N;

    if (sizeof(N) < sizeof A->dimensions[1]) {
        fprintf(stderr,
                "[linsolve::c_lapack] WARN Type `lapack_int` is smaller "
                "than the one used to describe dimensions of 'OIFArrayF64' "
                "variables\n");
    }

    if (A->dimensions[1] < INT_MAX) {
        N = (lapack_int)A->dimensions[1];
    }
    else {
        fprintf(stderr,
                "[linsolve::c_lapack] Dimensions of matrix are larger than "
                "this LAPACK implementation can handle\n");
        return 1;
    }

    lapack_int NRHS = 1;  // Number of right-hand sides.
    lapack_int LDA = N;   // Leading Dimension of A.
    lapack_int LDB = 1;   // Leading Dimension of b.

    int matrix_layout;

    if (OIF_ARRAY_C_CONTIGUOUS(A)) {
        matrix_layout = LAPACK_ROW_MAJOR;
    }
    else if (OIF_ARRAY_F_CONTIGUOUS(A)) {
        matrix_layout = LAPACK_COL_MAJOR;
        LDB = N;
    }
    else {
        fprintf(stderr,
                "[linsolve::c_lapack] Matrix A is not C or Fortran contiguous. Cannot proceed\n");
        return 1;
    }

    assert(NRHS == b->nd);
    assert(b->nd == x->nd);
    assert(b->dimensions[0] == x->dimensions[0]);

    double *Acopy = malloc(sizeof(double) * N * N);
    if (Acopy == NULL) {
        fprintf(stderr, "[linsolve::c_lapack] Could not allocate memory for matrix copy\n");
        return 2;
    }
    memcpy(Acopy, A->data, sizeof(double) * N * N);
    memcpy(x->data, b->data, sizeof(double) * N);

    fprintf(stderr, "[linsolve::c_lapack] A dimensions = %ld x %ld\n", A->dimensions[0],
            A->dimensions[1]);
    fprintf(stderr, "[linsolve::c_lapack] b dimensions: nd = %d, dim[0] = %ld\n", b->nd,
            b->dimensions[0]);

    int *ipiv = malloc(sizeof *ipiv * N);

    int info = LAPACKE_dgesv(matrix_layout, N, NRHS, Acopy, LDA, ipiv, x->data, LDB);

    if (info > 0) {
        fprintf(stderr, "[linsolve::c_lapack] LU factorization of A was not successfull\n");
        fprintf(stderr, "[linsolve::c_lapack] U(%i, %i) are zero, hence A is singular\n", info, info);
        goto cleanup;
    }
    else if (info < 0) {
        fprintf(stderr, "[linsolve::c_lapack] The %i-th argument had an illegal value\n", info);
        goto cleanup;
    }

cleanup:
    free(ipiv);
    free(Acopy);

    return info;
}

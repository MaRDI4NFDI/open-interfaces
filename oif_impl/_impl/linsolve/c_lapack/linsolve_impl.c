#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Use LAPACKE - C-friendly interface to LAPACK.
#include <lapacke.h>

#include <oif_impl/linsolve.h>

int
solve_lin(OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x)
{
    lapack_int N;

    if (sizeof(N) < sizeof A->dimensions[1]) {
        fprintf(stderr,
                "[c_lapack::solve_lin] WARN Type `lapack_int` is smaller "
                "than the one used to describe dimensions of 'OIFArrayF64' "
                "variables\n");
    }

    if (A->dimensions[1] < INT_MAX) {
        N = (lapack_int)A->dimensions[1];
    }
    else {
        fprintf(stderr,
                "[c_lapack::solve_lin] Dimensions of matrix are larger than "
                "this LAPACK implementation can handle\n");
        return 1;
    }

    lapack_int NRHS = 1;  // Number of right-hand sides.
    lapack_int LDA = N;   // Leading Dimension of A
    lapack_int LDB = 1;   // Leading Dimension of b

    assert(NRHS == b->nd);
    assert(b->nd == x->nd);
    assert(b->dimensions[0] == x->dimensions[0]);

    double *Acopy = malloc(sizeof(double) * N * N);
    if (Acopy == NULL) {
        fprintf(stderr, "[c_lapack:solve_lin] Could not allocate memory for matrix copy\n");
        return 2;
    }
    memcpy(Acopy, A->data, sizeof(double) * N * N);
    memcpy(x->data, b->data, sizeof(double) * N);

    fprintf(stderr, "[linsolve] A dimensions = %ld x %ld\n", A->dimensions[0],
            A->dimensions[1]);
    fprintf(stderr, "[linsolve] b dimensions: nd = %d, dim[0] = %ld\n", b->nd,
            b->dimensions[0]);

    int *ipiv = malloc(sizeof *ipiv * N);

    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, NRHS, Acopy, LDA, ipiv, x->data, LDB);

    if (info > 0) {
        fprintf(stderr, "[linsolve] LU factorization of A was not successfull\n");
        fprintf(stderr, "[linsolve] U(%i, %i) are zero, hence A is singular\n", info, info);
        goto cleanup;
    }
    else if (info < 0) {
        fprintf(stderr, "[linsolve] The %i-th argument had an illegal value", info);
        goto cleanup;
    }

cleanup:
    free(ipiv);
    free(Acopy);

    return info;
}

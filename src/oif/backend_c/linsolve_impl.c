#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Use LAPACKE - C-friendly interface to LAPACK.
#include <lapacke.h>

#include <oif/backend_c/linsolve.h>


int
solve_lin(OIFArrayF64 *A, OIFArrayF64 *b, OIFArrayF64 *x) {
    int N = A->dimensions[1];
    int NRHS = 1; // Number of right-hand sides.
    int LDA = N;  // Leading Dimension of A
    int LDB = 1;  // Leading Dimension of b
    
    assert(NRHS == b->nd);
    assert(b->nd == x->nd);
    assert(b->dimensions[0] == x->dimensions[0]);

    double *Acopy = malloc(N*N*sizeof(double))
    memcpy(Acopy, A->data, N*N*sizeof(double));
    memcpy(x->data, b->data, N*sizeof(double));

    fprintf(stderr, "[linsolve] A dimensions = %ld x %ld\n",
            A->dimensions[0], A->dimensions[1]);
    fprintf(stderr, "[linsolve] b dimensions: nd = %d, dim[0] = %ld\n",
            b->nd, b->dimensions[0]);
    
    int ipiv[N];

    int info = LAPACKE_dgesv(
        LAPACK_ROW_MAJOR, N, NRHS, Acopy, LDA, ipiv, x->data, LDB
    );

    if (info > 0) {
        fprintf(stderr, "[linsolve] LU factorization of A was not successfull\n");
        fprintf(
            stderr,
            "[linsolve] U(%i, %i) are zero, hence A is singular\n", info, info
        );
        free(Acopy);
        return 1;
    } else if (info < 0) {
        free(Acopy);
        fprintf(stderr, "[linsolve] The %i-th argument had an illegal value", info);
    }

    free(Acopy);
    return 0;
}

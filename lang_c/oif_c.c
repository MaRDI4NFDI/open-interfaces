#include "oif_config.h"
#include "oif_connector/oif_constants.h"

#include <malloc.h>
#include <math.h>
#include <memory.h>
#include <oif_connector/oif_interface.h>
#include <stdio.h>

#include <flexiblas/cblas.h>

int oif_lang_init() { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  fprintf(
      stderr,
      "Error: Evaluating string expression '%s' not possible in C connector\n",
      str);
  return OIF_NOT_IMPLEMENTED;
}

int oif_lang_deinit() { return OIF_OK; }

int oif_lang_solve(int N, double *A, double *b, double *x) {
  //! adjusted from https://cplusplus.com/forum/general/222617/
  double max_residual = 1.0e-6;
  double eps = 1e-10;

  double *R = calloc(N, sizeof(double));
  double *AP = calloc(N, sizeof(double));
  double *Rold = calloc(N, sizeof(double));
  double *P = calloc(N, sizeof(double));

  memcpy(P, R, N * sizeof(double));
  memcpy(R, b, N * sizeof(double));

  for (int in = 0; in < N; ++in) {
    x[in] = 0;
  }

  int k = 0;

  while (k < N) {
    memcpy(R, Rold, N * sizeof(double));

    cblas_dscal(N, 0.0, AP, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, P, 1, 0.0, AP, 1);

    double alpha =
        cblas_ddot(N, R, 1, R, 1) / fmax(cblas_ddot(N, R, 1, AP, 1), eps);

    // Next estimate of solution
    cblas_daxpy(N, alpha, P, 1, x, 1);
    // Residual
    cblas_daxpy(N, -alpha, AP, 1, R, 1);

    if (cblas_dnrm2(N, R, 1) < max_residual)
      break;

    double beta =
        cblas_ddot(N, R, 1, R, 1) / fmax(cblas_ddot(N, Rold, 1, Rold, 1), eps);
    // Next gradient
    cblas_dscal(N, beta, P, 1);
    cblas_daxpy(N, 1.0, R, 1, P, 1);
    k++;
  }

  free(R);
  free(Rold);
  free(AP);
  free(P);
  if (k < N)
    return OIF_OK;
  return OIF_RUNTIME_ERROR;
}

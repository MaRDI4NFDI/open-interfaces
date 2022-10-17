#include "oif_config.h"

extern "C" {
#include "oif_connector/oif_constants.h"
#include <oif_connector/oif_interface.h>
}
#include <array>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <flexiblas/cblas.h>

int oif_lang_init(void) { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  std::cerr << "Error: Evaluating string expression '" << str
            << "' not possible in Cpp connector" << std::endl;
  return OIF_NOT_IMPLEMENTED;
}

int oif_lang_deinit(void) { return OIF_OK; }

int oif_lang_solve(int N, double *A, double *b, double *x) {
  //! adjusted from https://cplusplus.com/forum/general/222617/
  using namespace std;
  constexpr double max_residual = 1.0e-6;
  constexpr double eps = numeric_limits<double>::epsilon();

  vector<double> R(b, b + N);
  assert(long(R.size()) == long(N));
  vector<double> AP(R.size(), 0.0);
  auto P = R;
  int k = 0;

  std::fill(x, x + N, 0.0);

  while (k < N) {
    auto Rold = R; // Store previous residual

    cblas_dscal(N, 0.0, AP.data(), 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, P.data(), 1, 0.0,
                AP.data(), 1);

    const double alpha = cblas_ddot(N, R.data(), 1, R.data(), 1) /
                         max(cblas_ddot(N, R.data(), 1, AP.data(), 1), eps);

    // Next estimate of solution
    cblas_daxpy(N, alpha, P.data(), 1, x, 1);
    // Residual
    cblas_daxpy(N, -alpha, AP.data(), 1, R.data(), 1);

    if (cblas_dnrm2(N, R.data(), 1) < max_residual)
      return OIF_OK;

    const double beta = cblas_ddot(N, R.data(), 1, R.data(), 1) /
                        max(cblas_ddot(N, Rold.data(), 1, Rold.data(), 1), eps);
    // Next gradient
    cblas_dscal(N, beta, P.data(), 1);
    cblas_daxpy(N, 1.0, R.data(), 1, P.data(), 1);
    k++;
  }

  return OIF_RUNTIME_ERROR;
}

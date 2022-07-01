#include "oif_config.h"

extern "C" {
#include "oif_connector/oif_constants.h"
#include <oif_connector/oif_interface.h>
}
#include <array>
#include <iostream>
#include <vector>

int oif_lang_init() { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  std::cerr << "Error: Evaluating string expression '" << str
            << "' not possible in Cpp connector" << std::endl;
  return OIF_NOT_IMPLEMENTED;
}

void oif_lang_deinit() {}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  //! adjusted from https://cplusplus.com/forum/general/222617/
  const double tol = 1.0e-6;

  std::vector<double> R(b, b + N);
  auto P = R;
  int k = 0;

  while (k < N) {
    auto Rold = R; // Store previous residual
    vec AP = matrixTimesVector(A, P);

    double alpha = innerProduct(R, R) / max(innerProduct(P, AP), NEARZERO);
    x = vectorCombination(1.0, x, alpha, P);   // Next estimate of solution
    R = vectorCombination(1.0, R, -alpha, AP); // Residual

    if (vectorNorm(R) < TOLERANCE)
      break; // Convergence test

    double beta = innerProduct(R, R) / max(innerProduct(Rold, Rold), NEARZERO);
    P = vectorCombination(1.0, R, beta, P); // Next gradient
    k++;
  }

  return x;
}

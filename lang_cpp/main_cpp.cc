extern "C" {
#include <oif_connector/oif_connector.h>
}

#include "oif_constants.h"
#include <string>

int main(int argc, char *argv[]) {
  std::string lang{"julia"};
  std::string expr{"print(42);"};
  if (argc > 2) {
    lang = std::string{argv[1]};
    expr = std::string{argv[2]};
  }

  oif_connector_init(lang.c_str());
  oif_connector_eval_expression(expr.c_str());

  constexpr int N{2};
  double A[N * N]{1, 0, 1, 0};
  double b[N]{1, 1};
  double x[N];

  if (oif_connector_solve(N, A, b, x) != OIF_OK) {
    return -1;
  }

  oif_connector_deinit();
}

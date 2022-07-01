#include <oif_connector/oif_connector.h>

#include "oif_constants.h"
#include <string.h>

int main(int argc, char *argv[]) {
  char *lang = "julia";
  char *expr = "print(42);";
  if (argc > 2) {
    lang = strdup(argv[1]);
    expr = strdup(argv[2]);
  }

  oif_connector_init(lang);
  oif_connector_eval_expression(expr);

  const int N = 2;
  double A[] = {1, 0, 1, 0};
  double b[] = {1, 1};
  double x[2];

  if (oif_connector_solve(N, A, b, x) != OIF_OK) {
    return -1;
  }
  oif_connector_deinit();
}

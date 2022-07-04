#include <oif_connector/oif_connector.h>

#include "oif_constants.h"
#include <stdio.h>
#include <string.h>

void print_vector(const int N, const double *x);

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
    print_vector(N, x);
    return -1;
  }
  print_vector(N, x);
  oif_connector_deinit();
}

void print_vector(const int N, const double *x) {
  printf("Solution: [");
  for (int i = 0; i < N; ++i) {
    printf("%f,", x[i]);
  }
  printf("]\n");
}

#include <oif_connector/oif_connector.h>

#include "oif_constants.h"
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_vector(const int N, const double *x);

START_TEST(test_all) {
  char *lang = "julia";
  char *expr = "print(42);";

  ck_assert(oif_connector_init(lang) == OIF_OK);

  const int N = 2;
  double A[] = {2, 0, 0, 1};
  double b[] = {1, 1};
  double x[2];

  ck_assert(oif_connector_solve(N, A, b, x) == OIF_OK);
  ck_assert(oif_connector_eval_expression(expr) == OIF_OK);
  ck_assert(oif_connector_deinit() == OIF_OK);
}

void print_vector(const int N, const double *x) {
  printf("Solution: [");
  for (int i = 0; i < N; ++i) {
    printf("%f,", x[i]);
  }
  printf("]\n");
}

Suite *test_all_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("Money");

  tc_core = tcase_create("Core");
  tcase_add_test(tc_core, test_all);
  suite_add_tcase(s, tc_core);

  return s;
}

int main(void) {

  int number_failed;
  Suite *s;
  SRunner *sr;

  s = test_all_suite();
  sr = srunner_create(s);

  srunner_run_all(sr, CK_NORMAL);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);
  return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

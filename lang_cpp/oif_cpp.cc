#include "oif_config.h"

extern "C" {
#include "oif_connector/oif_constants.h"
#include <oif_connector/oif_interface.h>
}
#include <iostream>

int oif_lang_init() { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  std::cerr << "Error: Evaluating string expression '" << str
            << "' not possible in Cpp connector" << std::endl;
  return OIF_NOT_IMPLEMENTED;
}

void oif_lang_deinit() {}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_RUNTIME_ERROR;
}

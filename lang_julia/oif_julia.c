#include "oif_connector/oif_constants.h"
#include <julia.h>
#include <oif_config.h>
#include <oif_connector/oif_interface.h>

int oif_lang_init() {
  jl_init();
  return OIF_OK;
}

int oif_lang_eval_expression(const char *str) {
  jl_eval_string(str);
  jl_flush_cstdio();
  return OIF_OK;
}

int oif_lang_deinit() {
  jl_atexit_hook(0);
  return OIF_OK;
}

int oif_lang_solve(int N, const double *const A, const double *const b,
                   double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_NOT_IMPLEMENTED;
}

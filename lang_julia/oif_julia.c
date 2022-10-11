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
  if (jl_exception_occurred()) {
    fprintf(stderr, "Julia String evaluation failed\n");
    jl_static_show((JL_STREAM *)stderr, jl_stderr_obj());
    fprintf(stderr, "\n");
    jl_static_show((JL_STREAM *)stderr, jl_exception_occurred());
    fprintf(stderr, "\n");
    jl_exception_clear();
    return OIF_RUNTIME_ERROR;
  }
  jl_flush_cstdio();
  jl_atexit_hook(0);
  return OIF_OK;
}

int oif_lang_deinit() {
  jl_atexit_hook(0);
  return OIF_OK;
}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_NOT_IMPLEMENTED;
}

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
  jl_value_t *errs = jl_stderr_obj();
  if (errs) {
    fprintf(stderr, "\nString eval threw an error: ");
    jl_static_show((JL_STREAM *)stderr, jl_current_exception());
    fprintf(stderr, "\n");
    jlbacktrace(); // written to STDERR_FILENO
    jl_value_t *showf = jl_get_function(jl_base_module, "show");
    if (showf != NULL) {
      jl_call2(showf, errs, jl_current_exception());
      fprintf(stderr, "\n");
    }
    return OIF_RUNTIME_ERROR;
  }

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

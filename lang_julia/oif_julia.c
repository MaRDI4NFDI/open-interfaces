#include <oif_config.h>

#include "oif_connector/oif_constants.h"
#include <julia.h>
#include <oif_connector/oif_interface.h>

jl_value_t *get_matrix_dims(int N);

int oif_lang_init(void) {
  jl_init();
  return OIF_OK;
}

int oif_lang_eval_expression(const char *str) {
  jl_eval_string(str);
  if (jl_exception_occurred()) {
    fprintf(stderr, "Julia String evaluation failed\n");
    jl_static_show((JL_STREAM *)stderr, jl_stderr_obj());
    fprintf(stderr, "\n");
    fprintf(stderr, "%s \n", jl_typeof_str(jl_exception_occurred()));
    fprintf(stderr, "\n");
    jl_exception_clear();
    return OIF_RUNTIME_ERROR;
  }
  jl_flush_cstdio();
  jl_atexit_hook(0);
  return OIF_OK;
}

int oif_lang_deinit(void) {
  jl_atexit_hook(0);
  return OIF_OK;
}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  jl_value_t *array_type =
      jl_apply_array_type((jl_value_t *)jl_float64_type, 1);
  jl_value_t *matrix_type =
      jl_apply_array_type((jl_value_t *)jl_float64_type, 2);
  jl_array_t *j_b = jl_ptr_to_array_1d(array_type, b, N, 0);
  jl_value_t *dims = get_matrix_dims(N);
  if (dims == NULL) {
    return OIF_RUNTIME_ERROR;
  }
  jl_array_t *j_A = jl_ptr_to_array(matrix_type, A, (jl_value_t *)dims, 0);
  assert(jl_array_dim(j_A, 0) == jl_array_dim(j_b, 0));
  assert(jl_array_dim(j_A, 1) == jl_array_dim(j_b, 0));

  jl_function_t *func = jl_get_function(jl_base_module, "\\");
  jl_array_t *ret =
      (jl_array_t *)jl_call2(func, (jl_value_t *)j_A, (jl_value_t *)j_b);
  if (jl_exception_occurred()) {
    fprintf(stderr, "\n%s \n", jl_typeof_str(jl_exception_occurred()));
    fprintf(stderr, "\n%s \n", jl_typeof_str(jl_current_exception()));
    jl_exception_clear();
    return OIF_RUNTIME_ERROR;
  }
  memcpy(x, (double *)jl_array_data(ret), N * sizeof(double));
  return OIF_OK;
}

jl_value_t *get_matrix_dims(int N) {
  /*
   * This should really not involve a string eval
   * but wrapping a 1D double array where each entry is `N`
   * and using jl_tupletype_fill(2, (jl_value_t*)jl_int32_type) and setting each
   * entry resulted in runtime errors or 0-dim matrix
   */
  char *dim_string;
  if (0 > asprintf(&dim_string, "(%d,%d)", N, N)) {
    return NULL;
  }
  return jl_eval_string(dim_string);
}

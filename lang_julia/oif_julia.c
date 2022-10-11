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
    fprintf(stderr, "%s \n", jl_typeof_str(jl_exception_occurred()));
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
  jl_value_t *int_array_type =
      jl_apply_array_type((jl_value_t *)jl_int64_type, 1);
  jl_value_t *array_type =
      jl_apply_array_type((jl_value_t *)jl_float64_type, 1);
  jl_value_t *matrix_type =
      jl_apply_array_type((jl_value_t *)jl_float64_type, 2);
  int dims[2];
  dims[0] = N;
  dims[1] = N;
  jl_array_t *j_dims = jl_ptr_to_array_1d(int_array_type, dims, 2, 0);
  jl_array_t *j_b = jl_ptr_to_array_1d(array_type, b, N, 0);
  jl_array_t *j_A = jl_ptr_to_array(matrix_type, A, (jl_value_t *)j_dims, 0);

  int size0 = jl_array_dim(j_A, 0);
  int size1 = jl_array_dim(j_A, 1);
  assert(size0 == N);
  assert(size1 == N);

  // Fill array with data
  for (int i = 0; i < size1; i++)
    for (int j = 0; j < size0; j++)
      printf("DIMS %d - %d", i, j);

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

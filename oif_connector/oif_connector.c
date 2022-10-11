#include "oif_connector.h"
#include "oif_constants.h"
#include <oif_config.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__oif_lib_handle;
char *__oif_current_lang;

#ifdef OIF_USE_R
#include <Rinternals.h>
#include <assert.h>

char *_R_convert_to_char(SEXP str) {
  if (!isString(str) || length(str) != 1) {
    return NULL;
  }
  return strdup(CHAR(STRING_ELT(str, 0)));
}
int oif_connector_init_r(SEXP lang) {
  const char *c_lang = _R_convert_to_char(lang);
  if (!c_lang)
    return OIF_TYPE_ERROR;
  return oif_connector_init(c_lang);
}
int oif_connector_eval_expression_r(SEXP str) {
  const char *c_str = _R_convert_to_char(str);
  if (!c_str)
    return OIF_TYPE_ERROR;
  return oif_connector_eval_expression(c_str);
}
void oif_connector_deinit_r() { oif_connector_deinit(); }
#else

int _no_R_error(const char *func) {
  fprintf(stderr, "Error: Calling R variant for %s not supported\n", func);
  return OIF_LOAD_ERROR;
}

int oif_connector_init_r(OIF_UNUSED SEXP lang) { return _no_R_error("init"); }
int oif_connector_eval_expression_r(OIF_UNUSED SEXP str) {
  return _no_R_error("lang_solve");
}
#endif

int oif_connector_init(const char *lang) {
  int (*init_lang)();
  __oif_current_lang = strdup(lang);

  char *libname;
  if (0 > asprintf(&libname, "lang_%s/liboif_%s.so", __oif_current_lang,
                   __oif_current_lang))
    return OIF_RUNTIME_ERROR;
  __oif_lib_handle = dlopen(libname, RTLD_LAZY);
  free(libname);

  if (!__oif_lib_handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    return OIF_LOAD_ERROR;
  }

  *(int **)(&init_lang) = dlsym(__oif_lib_handle, "oif_lang_init");
  if (!init_lang) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(__oif_lib_handle);
    return OIF_SYMBOL_ERROR;
  }

  return init_lang();
}

int oif_connector_eval_expression(const char *str) {
  int (*eval_expression)();
  *(void **)(&eval_expression) =
      dlsym(__oif_lib_handle, "oif_lang_eval_expression");
  if (!eval_expression) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(__oif_lib_handle);
    return OIF_SYMBOL_ERROR;
  }
  return eval_expression(str);
}

int oif_connector_deinit() {
  int ret = OIF_LOAD_ERROR;
  int (*lang_deinit)();
  *(int **)(&lang_deinit) = dlsym(__oif_lib_handle, "oif_lang_deinit");
  if (!lang_deinit) {
    fprintf(stderr, "Error: %s\n", dlerror());
  } else {
    ret = lang_deinit();
  }
  dlclose(__oif_lib_handle);
  free(__oif_current_lang);
  return ret;
}

int oif_connector_solve(int N, double *A, double *b, double *x) {
  assert(N > 0);
  assert(A);
  assert(b);
  assert(x);
  int (*lang_solve)();
  *(void **)(&lang_solve) = dlsym(__oif_lib_handle, "oif_lang_solve");
  if (!lang_solve) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(__oif_lib_handle);
    return OIF_SYMBOL_ERROR;
  }
  lang_solve(N, A, b, x);
  return OIF_OK;
}

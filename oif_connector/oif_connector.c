#include "oif_connector.h"
#include <oif_config.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__oif_lib_handle;
char *__oif_current_lang;

int oif_connector_init(const char *lang) {
  int (*init_lang)();
  __oif_current_lang = strdup(lang);

  char *libname;
  if (0 > asprintf(&libname, "liboif_%s.so", __oif_current_lang))
    return OIF_LOAD_ERROR;
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
  eval_expression(str);
  return OIF_OK;
}

void oif_connector_deinit() {
  dlclose(__oif_lib_handle);
  free(__oif_current_lang);
}

#include "oif_connector.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

void *__oif_lib_handle;

int oif_init_lang() {

  int (*init_lang)();

  __oif_lib_handle = dlopen("liboif_julia.so", RTLD_LAZY);

  if (!__oif_lib_handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    return OIF_LOAD_ERROR;
  }

  *(int **)(&init_lang) = dlsym(__oif_lib_handle, "oif_init_lang");
  if (!init_lang) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(__oif_lib_handle);
    return OIF_SYMBOL_ERROR;
  }

  init_lang();

  return EXIT_SUCCESS;
}

int oif_eval_expression(const char *str) {
  int (*eval_expression)();
  *(void **)(&eval_expression) = dlsym(__oif_lib_handle, "oif_eval_expression");
  if (!eval_expression) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(__oif_lib_handle);
    return OIF_SYMBOL_ERROR;
  }
  eval_expression(str);
}

void oif_deinit_lang() { dlclose(__oif_lib_handle); }

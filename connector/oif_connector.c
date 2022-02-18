#include "oif_connector.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int oif_init_lang() {
  void *handle;
  int (*init_lang)();

  handle = dlopen("liboif_julia.so", RTLD_LAZY);

  if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  *(int **)(&init_lang) = dlsym(handle, "oif_init_lang");
  if (!init_lang) {
    fprintf(stderr, "Error: %s\n", dlerror());
    dlclose(handle);
    return EXIT_FAILURE;
  }

  init_lang();
  dlclose(handle);

  return EXIT_SUCCESS;
}
void oif_eval_expression(const char *str) {}
void oif_deinit_lang() {}

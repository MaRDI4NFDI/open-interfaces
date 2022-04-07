#include "oif_config.h"

#include <oif_connector/oif_interface.h>
#include <stdio.h>

int oif_lang_init() {
  printf("Python connector not implemented");
  return OIF_LOAD_ERROR;
}

int oif_lang_eval_expression(OIF_UNUSED const char *str) {
  printf("Python connector not implemented");
  return OIF_SYMBOL_ERROR;
}

void oif_lang_deinit() { printf("Python connector not implemented"); }

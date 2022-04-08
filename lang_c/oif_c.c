#include "oif_config.h"

#include <oif_connector/oif_interface.h>
#include <stdio.h>

int oif_lang_init() { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  fprintf(
      stderr,
      "Error: Evaluating string expression '%s' not possible in C connector\n",
      str);
  return OIF_NOT_IMPLEMENTED;
}

void oif_lang_deinit() {}

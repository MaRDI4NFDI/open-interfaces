#include "oif_config.h"

extern "C" {
#include <oif_connector/oif_interface.h>
}
#include <iostream>

int oif_lang_init() { return OIF_OK; }

int oif_lang_eval_expression(const char *str) {
  std::cerr << "Error: Evaluating string expression '" << str
            << "' not possible in Cpp connector" << std::endl;
  return OIF_NOT_IMPLEMENTED;
}

void oif_lang_deinit() {}

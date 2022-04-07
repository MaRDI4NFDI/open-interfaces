#include <julia.h>
#include <oif_config.h>
#include <oif_connector/oif_interface.h>
// JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a
// shared library) if you want fast code.

int oif_lang_init() {
  jl_init();
  return 0;
}

int oif_lang_eval_expression(const char *str) {
  if (str)
    jl_eval_string(str);
  else
    jl_eval_string("print(sqrt(2.0))");
  return OIF_OK;
}

void oif_lang_deinit() { jl_atexit_hook(0); }

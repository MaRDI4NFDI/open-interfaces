#include "oif_julia.h"

#include <julia.h>
// JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a
// shared library) if you want fast code.

void oif_init_lang() { jl_init(); }

void oif_eval_expression(const char *str) {
  if (str)
    jl_eval_string(str);
  else
    jl_eval_string("print(sqrt(2.0))");
}

void oif_deinit_lang() { jl_atexit_hook(0); }

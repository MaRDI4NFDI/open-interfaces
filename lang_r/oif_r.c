#include "oif_r.h"
#include <Rembedded.h>
#include <Rinternals.h>
#include <oif_config.h>

int oif_init_lang() {
  int r_argc = 2;
  char *r_argv[] = {"R", "--verbose"};
  return Rf_initEmbeddedR(r_argc, r_argv);
}

int oif_eval_expression(const char *str) {
  if (str)
    0; // jl_eval_string(str);
  else
    1; // jl_eval_string("print(sqrt(2.0))");
  return OIF_OK;
}

void oif_deinit_lang() { Rf_endEmbeddedR(0); }

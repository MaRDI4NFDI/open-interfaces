#include "oif_r.h"
#include <R_ext/Parse.h>
#include <Rembedded.h>
#include <Rinternals.h>
#include <oif_config.h>

int oif_init_lang() {
  int r_argc = 2;
  char *r_argv[] = {"R", "--verbose"};
  return Rf_initEmbeddedR(r_argc, r_argv);
}

int oif_eval_expression(const char *str) {
  if (!str)
    str = "print(42)";
  SEXP cmdSexp, cmdexpr = R_NilValue;
  ParseStatus status;
  cmdSexp = PROTECT(allocVector(STRSXP, 1));
  SET_STRING_ELT(cmdSexp, 0, mkChar(str));
  cmdexpr = PROTECT(R_ParseVector(cmdSexp, -1, &status, R_NilValue));
  if (status != PARSE_OK) {
    UNPROTECT(2);
    error("invalid call %s", str);
  }
  /* Loop is needed here as EXPSEXP will be of length > 1 */
  for (int i = 0; i < length(cmdexpr); i++)
    eval(VECTOR_ELT(cmdexpr, i), R_GlobalEnv);
  UNPROTECT(2);

  return OIF_OK;
}

void oif_deinit_lang() { Rf_endEmbeddedR(0); }

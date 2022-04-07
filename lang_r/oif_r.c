#include <oif_connector/oif_interface.h>

// clang-format off
#include <Rembedded.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>
// clang-format on

#include <oif_config.h>

int oif_lang_init() {
  int r_argc = 2;
  char *r_argv[] = {"R", "--verbose"};
  return Rf_initEmbeddedR(r_argc, r_argv);
}

int oif_lang_eval_expression(const char *str) {
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

void oif_lang_deinit() { Rf_endEmbeddedR(0); }

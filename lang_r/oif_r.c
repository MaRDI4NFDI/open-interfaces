#include <oif_connector/oif_interface.h>

// clang-format off
#include <Rembedded.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>
// clang-format on

#include "oif_connector/oif_constants.h"
#include <Rinterface.h>
#include <oif_config.h>

int r_initialized = 0;

int oif_lang_init() {
  if (r_initialized > 0 || R_running_as_main_program) {
    return OIF_OK;
  }

  char *r_argv[] = {"R", "--vanilla", "--quiet"};
  const int r_argc = sizeof(r_argv) / sizeof(r_argv[0]);

  fprintf(stderr, "\ninitializaing R %d\n", r_initialized);
  const int res = Rf_initEmbeddedR(r_argc, r_argv);
  r_initialized += 1;
  // the embedded setup apparently always returns 1
  return res == 1 ? OIF_OK : OIF_LOAD_ERROR;
}

int oif_lang_eval_expression(const char *str) {
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

int oif_lang_deinit() {
  Rf_endEmbeddedR(0);
  r_initialized -= 1;
  return OIF_OK;
}

int oif_lang_solve(int N, const double *const A, const double *const b,
                   double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_NOT_IMPLEMENTED;
}

#include "oif_config.h"
#include "oif_connector/oif_constants.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <oif_connector/oif_interface.h>
#include <stdio.h>
// clang-format off
#include <oif_connector/disable_warnings.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <oif_connector/reenable_warnings.h>
// clang-format on

int __did_we_init_python = 0;

int oif_lang_init() {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
    __did_we_init_python = 1;
  }
  return OIF_OK;
}

int oif_lang_eval_expression(const char *str) {
  int ret = PyRun_SimpleString(str);
  return ret != 0 ? OIF_RUNTIME_ERROR : OIF_OK;
}

int oif_lang_deinit() {
  // TODO: use the _Ex version for sanity check
  if (__did_we_init_python)
    Py_Finalize();
  return OIF_OK;
}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_NOT_IMPLEMENTED;
}

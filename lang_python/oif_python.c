#include "oif_config.h"
#include "oif_connector/oif_constants.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <oif_connector/oif_interface.h>
#include <stdio.h>

int __did_we_init_python = 0;

int oif_lang_init() {
  if (!Py_IsInitialized()) {
    Py_InitializeEx(0);
    __did_we_init_python = 1;
  }
  return OIF_OK;
}

int oif_lang_eval_expression(const char *str) {
  PyObject *code = Py_CompileString(str, "test", Py_file_input);
  if (!code) {
    PyErr_Print();
    return OIF_RUNTIME_ERROR;
  }
  PyObject *main_module = PyImport_AddModule("__main__");
  if (!main_module) {
    PyErr_Print();
    return OIF_RUNTIME_ERROR;
  }
  PyObject *global_dict = PyModule_GetDict(main_module);
  if (!global_dict) {
    PyErr_Print();
    return OIF_RUNTIME_ERROR;
  }
  PyObject *local_dict = PyDict_New();
  if (!local_dict) {
    PyErr_Print();
    return OIF_RUNTIME_ERROR;
  }
  PyObject *eval = PyEval_EvalCode(code, global_dict, local_dict);

  return eval ? OIF_RUNTIME_ERROR : OIF_OK;
}

void oif_lang_deinit() {
  // TODO: use the _Ex version for sanity check
  if (__did_we_init_python)
    Py_Finalize();
}

int oif_lang_solve(int N, double *A, double *b, double *x) {
  (void)N;
  (void)A;
  (void)b;
  (void)x;
  return OIF_NOT_IMPLEMENTED;
}

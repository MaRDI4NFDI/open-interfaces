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
  // checks import internally
  import_array2("Failed to import numpy", OIF_RUNTIME_ERROR);
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
  long dim_N[1], dim_NN[2];
  dim_N[0] = N;
  dim_NN[0] = N;
  dim_NN[1] = N;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
  PyObject *np_b = PyArray_SimpleNewFromData(1, dim_N, NPY_DOUBLE, b);
  PyObject *np_A = PyArray_SimpleNewFromData(2, dim_NN, NPY_DOUBLE, A);
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic pop

  const char *module_name = "numpy.linalg";
  PyObject *pName = PyUnicode_FromString(module_name);
  PyObject *pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  // import function
  const char *func_name = "solve";
  PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
  PyArrayObject *np_ret =
      (PyArrayObject *)PyObject_CallFunctionObjArgs(pFunc, np_A, np_b, NULL);
  Py_DecRef(pFunc);
  Py_DecRef(pModule);
  const int ret = OIF_RUNTIME_ERROR ? np_ret == NULL : OIF_OK;
  if (ret == OIF_OK) {
    double *x_ret = (double *)PyArray_GETPTR1(np_ret, 0);
    memcpy(x, x_ret, N * sizeof(double));
  }
  Py_DecRef((PyObject *)np_ret);
  return ret;
}

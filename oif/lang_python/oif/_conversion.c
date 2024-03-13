#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <ffi.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#include "oif/api.h"

static PyObject *
numpy_array_from_oif_array_f64(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *retval = NULL;
    PyObject *ctypes_pointer;

    // We need to initialize PyArray_API (table of function pointers)
    // in every translation unit (separate .c file).
    // See the details in the accepted solution here:
    // https://stackoverflow.com/q/47026900/1095202
    if (PyArray_API == NULL) {
        import_array();
    }

    if (!PyArg_ParseTuple(args, "O", &ctypes_pointer)) {
        fprintf(stderr, "[_conversion] Could not parse function arguments\n");
        return NULL;
    }

    // Life is too short not to play with raw pointers.
    OIFArrayF64 *arr = PyLong_AsVoidPtr(ctypes_pointer);
    if (arr == NULL) {
        fprintf(stderr,
                "[_conversion] Could not convert pointer "
                "to OIFArrayF64 data structure\n");
        return NULL;
    }

    retval = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64, arr->data);

    if (retval == NULL) {
        fprintf(stderr, "[_conversion] Could not create a new NumPy array\n");
    }

    return retval;
}

static PyMethodDef callback_methods[] = {
    {"numpy_array_from_oif_array_f64", numpy_array_from_oif_array_f64, METH_VARARGS,
     "Invoke a given C function from Python."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyDoc_STRVAR(conversion_doc, "Conversion functions between C and Python types");

static struct PyModuleDef callbackmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_conversion", /* name of module */
    .m_doc = conversion_doc, /* module documentation, may be NULL */
    .m_size = -1,            /* size of per-interpreter state of the module,
                                or -1 if the module keeps state in global variables. */
    .m_methods = callback_methods};

PyMODINIT_FUNC
PyInit__conversion(void)
{
    PyObject *m;

    m = PyModule_Create(&callbackmodule);
    if (m == NULL) {
        return NULL;
    }

    return m;
}

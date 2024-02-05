#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <ffi.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "oif/api.h"

static PyObject *call_c_fn_from_python(PyObject *self, PyObject *args) {
    PyObject *capsule;
    PyObject *py_args;

    // We need to initialize PyArray_API (table of function pointers)
    // in every translation unit (separate .c file).
    // See the details in the accepted solution here:
    // https://stackoverflow.com/q/47026900/1095202
    import_array();

    if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyTuple_Type, &py_args)) {
        fprintf(stderr, "[_callback] Could not parse function arguments");
        return NULL;
    }

    unsigned int nargs = 3;

    ffi_cif cif;
    ffi_type **arg_types = malloc(nargs * sizeof(ffi_type *));
    if (arg_types == NULL) {
        fprintf(stderr,
                "[_callback] Could not allocate memory for `arg_types`");
        return NULL;
    }
    void **arg_values = malloc(nargs * sizeof(void *));
    if (arg_values == NULL) {
        fprintf(stderr,
                "[_callback] Could not allocate memory for `arg_values`");
        free(arg_types);
        return NULL;
    }

    int arg_type_ids[] = {OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64};
    Py_ssize_t n_args = sizeof arg_type_ids / sizeof arg_type_ids[0];
    void *fn_p = PyCapsule_GetPointer(capsule, "123");
    printf("Function pointer is %p\n", fn_p);

    assert(n_args == PyTuple_Size(py_args));
    // Merge input and output argument types together in `arg_types` array.
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject *arg = PyTuple_GetItem(py_args, i);
        if (arg_type_ids[i] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
            if (!PyFloat_Check(arg)) {
                fprintf(stderr, "Expected PyFloat object.\n");
            }
            double double_value = PyFloat_AsDouble(arg);
            printf("Received double value: %f\n", double_value);
            arg_values[i] = &double_value;
        } else if (arg_type_ids[i] == OIF_ARRAY_F64) {
            arg_types[i] = &ffi_type_pointer;
            OIFArrayF64 array_value = {
                .nd = PyArray_NDIM((PyArrayObject *)arg),
                .dimensions = PyArray_DIMS((PyArrayObject *)arg),
                .data = PyArray_DATA((PyArrayObject *)arg)};
            arg_values[i] = &array_value;
        } else {
            fflush(stdout);
            fprintf(stderr,
                    "[dispatch_c] Unknown input arg type: %d\n",
                    arg_type_ids[i]);
            exit(EXIT_FAILURE);
        }
    }

    ffi_status status = ffi_prep_cif(
        &cif, FFI_DEFAULT_ABI, nargs, &ffi_type_sint, arg_types);
    if (status != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[dispatch_c] ffi_prep_cif was not OK");
        exit(EXIT_FAILURE);
    }

    int result;
    ffi_call(&cif, FFI_FN(fn_p), &result, arg_values);

    free(arg_values);
    free(arg_types);

    return PyLong_FromLong(result);
}

static PyMethodDef callback_methods[] = {
    {"call_c_fn_from_python",
     call_c_fn_from_python,
     METH_VARARGS,
     "Invoke a given C function from Python."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyDoc_STRVAR(
    callback_doc,
    "This module contains a function to invoke a given C function from Python "
    "converting NumPy arrays to their OIFArrayF64 analogs.");

static struct PyModuleDef callbackmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "callback",  /* name of module */
    .m_doc = callback_doc, /* module documentation, may be NULL */
    .m_size = -1,          /* size of per-interpreter state of the module,
                              or -1 if the module keeps state in global variables. */
    .m_methods = callback_methods};

PyMODINIT_FUNC PyInit__callback(void) {
    PyObject *m;

    m = PyModule_Create(&callbackmodule);
    if (m == NULL) {
        return NULL;
    }

    return m;
}

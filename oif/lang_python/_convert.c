/**
 * C library that converts OIF data types to Python datatypes.
 * Note that this library is not a Python extension module -
 * it uses Python C API but only to convert to Python.
 */
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <oif/api.h>


void
python_types_from_oif_types(PyObject *py_args, void **c_args, int *arg_types)
{
    /* // We need to initialize PyArray_API (table of function pointers) */
    /* // in every translation unit (separate .c file). */
    /* // See the details in the accepted solution here: */
    /* // https://stackoverflow.com/q/47026900/1095202 */
    /* if (PyArray_API == NULL) { */
    /*     import_array(); */
    /* } */

    size_t nargs = PyList_Size(py_args);
    PyObject *cur_arg = NULL;
    for (size_t i = 0; i < nargs; ++i) {
        if (arg_types[i] == OIF_INT) {
            cur_arg = PyLong_FromLong(*(int *) c_args[i]);
        }
        else if (arg_types[i] == OIF_FLOAT64) {
            cur_arg = PyFloat_FromDouble(*(double *) c_args[i]);
        }
        else if (arg_types[i] == OIF_ARRAY_F64) {
            OIFArrayF64 *arr = (OIFArrayF64 *) c_args[i];
            cur_arg = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64, arr->data);
            if (cur_arg == NULL) {
                fprintf(stderr, "[_convert] Could not create a new NumPy array\n");
            }
        }
        else if (arg_types[i] == OIF_USER_DATA) {
            cur_arg = (PyObject *) c_args[i];
        }
        else {
            fprintf(
                stderr,
                "[_convert] Cannot convert argument\n"
            );
        }
        PyList_SetItem(py_args, i, cur_arg);
    }
}

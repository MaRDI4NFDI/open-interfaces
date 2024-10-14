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

static bool INITIALIZED_ = false;

static PyObject *addressof_fn = NULL;

static void
init_(void)
{
    PyObject *ctypes = PyImport_ImportModule("ctypes");
    addressof_fn = PyObject_GetAttrString(ctypes, "addressof");

    // We need to initialize PyArray_API (table of function pointers)
    // in every translation unit (separate .c file).
    // See the details in the accepted solution here:
    // https://stackoverflow.com/q/47026900/1095202
    if (PyArray_API == NULL) {
        import_array();
    }

    INITIALIZED_ = true;
}


int
python_types_from_oif_types(PyObject *py_args, PyObject *c_args, PyObject *arg_types)
{
    /* // We need to initialize PyArray_API (table of function pointers) */
    /* // in every translation unit (separate .c file). */
    /* // See the details in the accepted solution here: */
    /* // https://stackoverflow.com/q/47026900/1095202 */
    /* if (PyArray_API == NULL) { */
    /*     import_array(); */
    /* } */

    PyGILState_STATE gstate;

    // Ensure we have the GIL
    gstate = PyGILState_Ensure();

    if (! INITIALIZED_) {
        init_();
    }

    size_t nargs = PyList_Size(py_args);
    PyObject *cur_arg = NULL;
    for (size_t i = 0; i < nargs; ++i) {
        PyObject *py_type = PyTuple_GET_ITEM(arg_types, i);
        long c_type = PyLong_AsLong(py_type);
        PyObject *py_cur_arg = PyTuple_GET_ITEM(c_args, i);
        if (c_type == OIF_INT) {
            cur_arg = py_cur_arg;
        }
        else if (c_type == OIF_FLOAT64) {
            cur_arg = py_cur_arg;
        }
        else if (c_type == OIF_ARRAY_F64) {
            if (! PyObject_HasAttrString(py_cur_arg, "contents")) {
                fprintf(
                    stderr,
                    "[_convert] Pass OIF_ARRAY_F64 is not ctypes object\n"
                );
                exit(EXIT_FAILURE);
            }

            PyObject *oif_array_f64_obj = PyObject_GetAttrString(py_cur_arg, "contents");
            PyObject *py_nd = PyObject_GetAttrString(oif_array_f64_obj, "nd");
            size_t nd = PyLong_AsLong(py_nd);
            assert(nd == 1);

            PyObject *py_dims = PyObject_GetAttrString(oif_array_f64_obj, "dimensions");
            PyObject *py_dimensions = PyObject_GetAttrString(py_dims, "contents");
            PyObject *py_dimensions_ptr = PyObject_GetAttrString(py_dimensions, "value");
            intptr_t dims = PyLong_AsLong(py_dimensions_ptr);
            intptr_t dimensions[] = {dims};

            PyObject *py_data_obj = PyObject_GetAttrString(oif_array_f64_obj, "data");
            PyObject *py_data_ptr = PyObject_GetAttrString(py_data_obj, "contents");
            PyObject *addressof_args = Py_BuildValue("(O)", py_data_ptr);
            PyObject *ptr_as_py_long = PyObject_CallObject(addressof_fn, addressof_args);
            double *data = PyLong_AsVoidPtr(ptr_as_py_long);

            printf("data: %p\n", data);

            cur_arg = PyArray_SimpleNewFromData(nd, dimensions, NPY_FLOAT64, data);
            if (cur_arg == NULL) {
                fprintf(stderr, "[_convert] Could not create a new NumPy array\n");
            }
        }
        else if (c_type == OIF_USER_DATA) {
            cur_arg = (PyObject *) PyLong_AsVoidPtr(py_cur_arg);
        }
        else {
            fprintf(
                stderr,
                "[_convert] Cannot convert argument\n"
            );
        }
        PyList_SetItem(py_args, i, cur_arg);
    }
finally:
    // Release the GIL
    PyGILState_Release(gstate);

    return 0;
}

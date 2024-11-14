/**
 * C library that converts OIF data types to Python datatypes.
 * Note that this library is not a Python extension module -
 * it uses Python C API but only to convert to Python.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <oif/api.h>

static bool INITIALIZED_ = false;

static PyObject *addressof_fn = NULL;

static double ELAPSED_ = 0;
static double ELAPSED_FN_ = 0;

void __attribute__((destructor))
dtor();

static int
init_(void)
{
    PyGILState_STATE gstate;

    // Ensure we have the GIL
    gstate = PyGILState_Ensure();
    PyObject *ctypes = PyImport_ImportModule("ctypes");
    addressof_fn = PyObject_GetAttrString(ctypes, "addressof");

    // We need to initialize PyArray_API (table of function pointers)
    // in every translation unit (separate .c file).
    // See the details in the accepted solution here:
    // https://stackoverflow.com/q/47026900/1095202
    if (PyArray_API == NULL) {
        import_array1(-1);
    }

    PyGILState_Release(gstate);

    INITIALIZED_ = true;
    return 0;
}

void
init(void)
{
    fprintf(stderr, "[_convert] init function\n");
    init_();
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
    fprintf(stderr, "I am in conversion function\n");

    PyGILState_STATE gstate;

    // Ensure we have the GIL
    gstate = PyGILState_Ensure();

    if (!INITIALIZED_) {
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
            if (!PyObject_HasAttrString(py_cur_arg, "contents")) {
                fprintf(stderr, "[_convert] Pass OIF_ARRAY_F64 is not ctypes object\n");
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

            cur_arg = PyArray_SimpleNewFromData(nd, dimensions, NPY_FLOAT64, data);
            if (cur_arg == NULL) {
                fprintf(stderr, "[_convert] Could not create a new NumPy array\n");
            }
        }
        else if (c_type == OIF_USER_DATA) {
            if (py_cur_arg == Py_None) {
                cur_arg = Py_None;
                Py_INCREF(Py_None);
            }
            else {
                cur_arg = (PyObject *)PyLong_AsVoidPtr(py_cur_arg);
            }
        }
        else {
            fprintf(stderr, "[_convert] Cannot convert argument\n");
        }
        if (cur_arg == NULL) {
            fprintf(stderr, "[_convert] cur_arg is NULL\n");
            PyErr_Print();
        }
        PyList_SetItem(py_args, i, cur_arg);
    }

    if (PyErr_Occurred()) {
        fprintf(stderr, "[_convert] Error occurred\n");
        PyErr_Print();
    }
finally:
    // Release the GIL
    PyGILState_Release(gstate);

    return 0;
}

int
c_wrapper_over_py_callable(void *py_fn, OIFArgs *args)
{
    clock_t tic = clock();
    size_t num_args = args->num_args;
    PyObject *py_args = PyTuple_New(num_args);
    PyObject *cur_arg = NULL;

    if (!PyCallable_Check((PyObject *)py_fn)) {
        fprintf(stderr, "[_convert] It is not a Python callable\n");
        return -1;
    }

    OIFArrayF64 *arr = NULL;

    for (size_t i = 0; i < num_args; ++i) {
        void *c_arg = args->arg_values[i];
        switch (args->arg_types[i]) {
            case OIF_INT:
                cur_arg = PyLong_FromLong(*(int *)c_arg);
                break;
            case OIF_FLOAT64:
                cur_arg = PyFloat_FromDouble(*(double *)c_arg);
                break;
            case OIF_ARRAY_F64:
                arr = *(OIFArrayF64 **)c_arg;
                assert(arr->nd == 1);
                cur_arg = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64,
                                                    arr->data);
                break;
            case OIF_USER_DATA:
                cur_arg = Py_None;
                Py_INCREF(Py_None);
                /* cur_arg = c_arg; */
                /* if (cur_arg == Py_None) { */
                /*     Py_INCREF(Py_None); */
                /* } */
                /* else { */
                /*     fprintf(stderr, "[_convert] Currently, the case with USER_DATA is not
                 * handled\n"); */
                /*     exit(1); */
                /* } */
                break;
            default:
                fprintf(stderr, "[_convert] BAD\n");
        }
        if (cur_arg == NULL) {
            fprintf(stderr, "[_convert] It did not work\n");
            Py_DECREF(py_args);
            return -1;
        }

        PyTuple_SetItem(py_args, i, cur_arg);
    }
    clock_t toc = clock();
    ELAPSED_ += toc - tic;

    tic = clock();
    PyObject *result = PyObject_CallObject(py_fn, py_args);
    Py_DECREF(py_args);
    toc = clock();
    ELAPSED_FN_ = toc - tic;

    if (result == NULL) {
        fprintf(stderr, "[_convert] Function invokation was bad\n");
        Py_DECREF(result);
        return 2;
    }
    Py_DECREF(result);
    return 0;
}

void
dtor(void)
{
    printf("Elapsed time in conversion: %.3f seconds\n", ELAPSED_ / CLOCKS_PER_SEC);
    printf("Elapsed time in  func call: %.3f seconds\n", ELAPSED_FN_ / CLOCKS_PER_SEC);
}

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <ffi.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "oif/api.h"

static PyObject *call_c_fn_from_python(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *retval = NULL;
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

    Py_ssize_t nargs_s = PyTuple_Size(py_args);
    if (nargs_s < 0) {
        fprintf(
            stderr,
            "[_callback] Unexpected negative value of the tuple of args "
            "for the callback function\n"
        );
        return NULL;
    }
    unsigned int nargs;
    if (nargs_s <= UINT_MAX) {
        nargs = (unsigned int) nargs_s;  // Explicit cast to eliminate compiler warning
    } else {
        fprintf(
            stderr,
            "[_callback] Could not convert size of the tuple of args "
            "to 'unsigned int' type\n"
        );
        return NULL;
    }
    int arg_type_ids[] = {OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64};

    ffi_cif cif;
    ffi_type **arg_types = malloc(nargs * sizeof(ffi_type *));
    if (arg_types == NULL) {
        fprintf(stderr,
                "[_callback] Could not allocate memory for `arg_types`\n");
        return NULL;
    }

    // We need to allocate memory for all values, to make sure
    // that the lifetime of the arguments ends after `ffi_call` below.
    void **arg_values = calloc(nargs, sizeof(void *));
    if (arg_values == NULL) {
        fprintf(stderr,
                "[_callback] Could not allocate memory for `arg_values`\n");
        goto clean_arg_types;
    }
    int narray_args = 0;  // Number of arguments of type `OIFArrayF64 *`.
    for (size_t i = 0; i < nargs; ++i) {
        if (arg_type_ids[i] == OIF_INT) {
            arg_values[i] = malloc(sizeof(int));
        } else if (arg_type_ids[i] == OIF_FLOAT64) {
            arg_values[i] = malloc(sizeof(double));
        } else if (arg_type_ids[i] == OIF_ARRAY_F64) {
            arg_values[i] = malloc(sizeof(OIFArrayF64 **));
            narray_args++;
        } else {
            fprintf(stderr,
                    "[_callback] Unknown input arg type: %d\n",
                    arg_type_ids[i]);
            goto clean_arg_values;
        }
        if (arg_values[i] == NULL) {
            fprintf(stderr,
                    "[_callback] Could not allocate memory for element #%zu\n",
                    i);
            goto clean_arg_values;
        }
    }
    
    OIFArrayF64 **oif_arrays = calloc(narray_args, sizeof(OIFArrayF64 *));
    if (oif_arrays == NULL) {
        fprintf(stderr,
                "[_callback] Could not allocate memory for `oif_arrays`\n");
        goto clean_arg_values;
    }
    for (Py_ssize_t i = 0; i < narray_args; ++i) {
        oif_arrays[i] = malloc(sizeof(OIFArrayF64));
        if (oif_arrays[i] == NULL) {
            fprintf(stderr,
                    "[_callback] Could not allocate memory for `oif_arrays[%ld]`\n",
                    i);
            goto clean_oif_arrays;
        }
    }

    void *fn_p = PyCapsule_GetPointer(capsule, "123");

    // Prepare function arguments for FFI expectations (pointers)
    // and convert NumPy arrays to OIFArrayF64 structs.
    for (size_t i = 0, j = 0; i < nargs; ++i) {
        PyObject *arg = PyTuple_GetItem(py_args, i);
        if (arg_type_ids[i] == OIF_FLOAT64) {
            arg_types[i] = &ffi_type_double;
            if (!PyFloat_Check(arg)) {
                fprintf(stderr, "[_callback] Expected PyFloat object.\n");
                goto clean_arg_values;
            }
            double *double_value = arg_values[i];
            *double_value = PyFloat_AsDouble(arg);
        } else if (arg_type_ids[i] == OIF_ARRAY_F64) {
            arg_types[i] = &ffi_type_pointer;
            PyArrayObject *py_arr = (PyArrayObject *) arg;
            if (!PyArray_Check(py_arr)) {
                fprintf(stderr, "[_callback] Expected PyArrayObject (NumPy ndarray) object\n");
                goto clean_arg_values;
            }
            oif_arrays[j]->nd =PyArray_NDIM(py_arr); 
            oif_arrays[j]->dimensions = PyArray_DIMS(py_arr);
            oif_arrays[j]->data = PyArray_DATA(py_arr);
            // We always pass array data structure as pointer: `OIFArrayF64 *`,
            // and FFI requires pointer to function arguments;
            // hence, we need to obtain `OIFArrayF64 **`.
            OIFArrayF64 **pp = arg_values[i];
            *pp = oif_arrays[j];
            j++;
        } else {
            fflush(stdout);
            fprintf(stderr,
                    "[_callback] Unknown input arg type: %d\n",
                    arg_type_ids[i]);
            goto clean_arg_values;
        }
    }

    ffi_status status = ffi_prep_cif(
        &cif, FFI_DEFAULT_ABI, nargs, &ffi_type_sint, arg_types);
    if (status != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[_callback] ffi_prep_cif was not OK");
        goto clean_arg_values;
    }

    int result;
    ffi_call(&cif, FFI_FN(fn_p), &result, arg_values);

    retval = PyLong_FromLong(result);
clean_arg_values:
    free(arg_values);
clean_arg_types:
    free(arg_types);
    return retval;
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

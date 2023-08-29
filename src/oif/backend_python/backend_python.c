#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <dlfcn.h>
#include <ffi.h>

#include "dispatch.h"
#include "globals.h"


BackendHandle load_backend(
    const char *operation,
    size_t version_major,
    size_t version_minor)
{
    if (Py_IsInitialized()) {
        fprintf(stderr, "[backend_python] Backend is already initialized\n");
    } else {
        PyConfig py_config;
        PyConfig_InitPythonConfig(&py_config);
    }

    PyRun_SimpleString(
        "import sys; "
        "print('[backend_python]', sys.executable); "
        "print('[backend_python]', sys.version)"
    );

    import_array2(
        "Failed to initialize NumPy C API",
        OIF_ERROR
    );

    return BACKEND_PYTHON;
}


int run_interface_method(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    PyObject *pFileName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    printf("[backend_python] Provided module name: %s\n", method);
    pFileName = PyUnicode_FromString(method);

    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Failed to load \"%s\"\n", method);
        return EXIT_FAILURE;
    }

    pFunc = PyObject_GetAttrString(pModule, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        int num_args = in_args->num_args + out_args->num_args;
        pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (int i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            } else if (in_args->arg_types[i] == OIF_FLOAT64_P) {
                OIFArray *arr = (OIFArray *) in_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, *(double **) arr->data
                );
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr, "[backend_python] Cannot convert argument\n");
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (int i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            } else if (out_args->arg_types[i] == OIF_FLOAT64_P) {
                OIFArray *arr = *(OIFArray **) out_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, (double *) arr->data
                );
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr,
                        "[backend_python] Cannot convert out_arg %d of type %d\n",
                        i, out_args->arg_types[i]);
                return 1;
            }
            PyTuple_SetItem(pArgs, i + in_args->num_args, pValue);
        }

        // Invoke function.
        pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (pValue != NULL) {
            printf("Status of call from C to Python: %ld\n", PyLong_AsLong(pValue));
            Py_DECREF(pValue);
        } else {
            Py_DECREF(pFunc);
            Py_DECREF(pModule);
            PyErr_Print();
            fprintf(stderr, "Call failed\n");
            return EXIT_FAILURE;
        }
    } else {
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
        fprintf(stderr, "Cannot find function \"%s\"\n", method);
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);

    return 0;
}

int unload_backend_python() {
    if (!Py_IsInitialized()) {
        return 0;
    }

    if (Py_FinalizeEx() < 0) {
        return 120;
    }

    return 0;
}

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <ffi.h>

#include "dispatch.h"
#include "globals.h"

#include <numpy/arrayobject.h>


int run_interface_method_python(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    PyObject *pFileName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    
    if (Py_IsInitialized()) {
        fprintf(stderr, "already initialized\n");
    }

    PyConfig py_config;
    PyConfig_InitPythonConfig(&py_config);

    system("which python");
    PyRun_SimpleString("import sys; sys.__version__");
    printf("Provided module name: %s\n", method);
    pFileName = PyUnicode_FromString(method);
    printf("PyUnicode_FromString module name: %s\n", PyUnicode_AsUTF8(pFileName));

    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", method);
        return EXIT_FAILURE;
    }

    import_array2(
        "Failed to initialize NumPy C API",
        OIF_ERROR
    );

    pFunc = PyObject_GetAttrString(pModule, method);
    /* pFunc is a new reference */
    double *data;

    if (pFunc && PyCallable_Check(pFunc)) {
        int num_args = in_args->num_args + out_args->num_args;
        pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (int i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            } else if (in_args->arg_types[i] == OIF_FLOAT64_P) {
                const long shape[] = {2};
                pValue = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, in_args->arg_values[i]);
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr, "Cannot convert argument\n");
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (int i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            } else if (out_args->arg_types[i] == OIF_FLOAT64_P) {
                fprintf(stderr, "Oh Gott! Remove hardcoded values ASAP\n");
                npy_intp shape[] = {2};
                void **p = out_args->arg_values[i];
                data = *p;
                Py_INCREF(data);
                pValue = PyArray_SimpleNewFromData(1, shape, NPY_FLOAT64, data);
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

    if (Py_FinalizeEx() < 0) {
        return 120;
    }

    return 0;
}

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <dlfcn.h>
#include <ffi.h>

#include <oif/api.h>
#include <oif/dispatch_api.h>

typedef struct {
    PyObject *pInstance;
} PythonImpl;

static PythonImpl *IMPL;


BackendHandle load_backend(
    const char *impl_details,
    size_t version_major,
    size_t version_minor)
{
    if (Py_IsInitialized()) {
        fprintf(stderr, "[backend_python] Backend is already initialized\n");
    } else {
        Py_Initialize();
    }

    // We need to `dlopen` the Python library, otherwise,
    // NumPy initialization fails.
    // Details: https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    char libpython_name[32];
    sprintf(libpython_name, "libpython%d.%d.so", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    void *libpython = dlopen(libpython_name, RTLD_LAZY | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "[backend_python] Cannot open python library\n");
        exit(EXIT_FAILURE);
    }

    PyRun_SimpleString(
        "import sys; "
        "print('[backend_python]', sys.executable); "
        "print('[backend_python]', sys.version)"
    );

    import_array2(
        "Failed to initialize NumPy C API",
        OIF_BACKEND_INIT_ERROR
    );

    PyRun_SimpleString(
        "import numpy; "
        "print('[backend_python] NumPy version: ', numpy.__version__)"
    );

    char moduleName[512] = "\0";
    char className[512] = "\0";
    size_t i;
    for (i = 0; i < strlen(impl_details); ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            moduleName[i] = impl_details[i];
        } else {
            moduleName[i] = '\0';
            break;
        }
    }
    size_t offset = i + 1;
    for (; i < strlen(impl_details); ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            className[i - offset] = impl_details[i];
        } else {
            className[i] = '\0';
        }
    }

    PyObject *pFileName, *pModule;
    PyObject *pClass, *pInstance;
    PyObject *pInitArgs; 
    printf("[backend_python] Provided module name: '%s'\n", moduleName);
    printf("[backend_python] Provided class name: '%s'\n", className);
    pFileName = PyUnicode_FromString(moduleName);
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(
            stderr,
            "[backend_python] Failed to load module \"%s\". ",
            moduleName
        );
        return EXIT_FAILURE;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        fprintf(
            stderr,
            "[backend_python] Failed to instantiate class %s\n",
            className
        );
        Py_DECREF(pClass);
        return EXIT_FAILURE;
    }
    Py_DECREF(pClass);

    IMPL = malloc(sizeof(PythonImpl));
    IMPL->pInstance = pInstance;

    return BACKEND_PYTHON;
}


int run_interface_method(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    PyObject *pFunc;
    PyObject *pArgs;
    PyObject *pValue;

    pFunc = PyObject_GetAttrString(IMPL->pInstance, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        int num_args = in_args->num_args + out_args->num_args;
        pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (int i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            } else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **) in_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, arr->data
                );
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pFunc);
                fprintf(stderr, "[backend_python] Cannot convert input argument #%d\n", i);
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (int i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            } else if (out_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **) out_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, arr->data
                );
            }
            if (!pValue) {
                Py_DECREF(pArgs);
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
            Py_DECREF(pValue);
        } else {
            Py_DECREF(pFunc);
            PyErr_Print();
            fprintf(stderr, "Call failed\n");
            return EXIT_FAILURE;
        }
    } else {
        if (PyErr_Occurred()) {
            fprintf(stderr, "[dispatch_python] ");
            PyErr_Print();
        }
        fprintf(stderr, "[dispatch_python] Cannot find function \"%s\"\n", method);
        return -1;
    }
    Py_XDECREF(pFunc);

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

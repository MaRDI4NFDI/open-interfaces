#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <dlfcn.h>
#include <ffi.h>
#include <stdlib.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/dispatch_api.h>

typedef struct {
    ImplInfo base;
    PyObject *pInstance;
} PythonImplInfo;

PyObject *CALLBACK_CLASS_P = NULL;

static int IMPL_COUNTER = 0;

typedef void (*ivp_rhs_fp_t)(double t, OIFArrayF64 *y, OIFArrayF64 *ydot);
ivp_rhs_fp_t IVP_RHS_CALLBACK = NULL;

int instantiate_callback_class() {
    char *moduleName = "oif.callback";
    char class_name[] = "Callback";

    PyObject *pFileName = PyUnicode_FromString(moduleName);
    PyObject *pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Faile to load callback module\n");
        exit(1);
    }
   
    CALLBACK_CLASS_P = PyObject_GetAttrString(pModule, class_name);
    if (CALLBACK_CLASS_P == NULL) {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Cannot proceed as callback class %s could not be instantiated\n", class_name);
    }
    Py_DECREF(pModule);

    return 0;
}

PyObject *convert_oif_callback_struct_to_pyobject(OIFCallback *p) {
    const char *id = "123";
    PyObject *fn_p = PyCapsule_New(p->fn_p_c, id, NULL);
    if (fn_p == NULL) {
        fprintf(stderr, "[dispatch_python] Could not create PyCapsule\n");
    }
    PyObject *obj = Py_BuildValue(
        "(O, s)", fn_p, id
    );
    if (obj == NULL) {
        fprintf(stderr, "[backend_python] Could not build arguments\n");
    }
    return obj;
}

static PyObject *c_to_py_wrapper(PyObject *ignored, PyObject *args) {
    double t; // Time
    PyArrayObject *y_ndarray;
    PyArrayObject *ydot_ndarray;

    PyArrayObject **arg_values;

    if (!PyArg_ParseTuple(args,
                          "dO!O!",
                          &t,
                          &PyArray_Type,
                          &y_ndarray,
                          &PyArray_Type,
                          &ydot_ndarray)) {
        return NULL;
    }

    OIFArrayF64 y = {
        .nd = PyArray_NDIM(y_ndarray),
        .dimensions = PyArray_DIMS(y_ndarray),
        .data = PyArray_DATA(y_ndarray),
    };

    OIFArrayF64 y_dot = {
        .nd = PyArray_NDIM(ydot_ndarray),
        .dimensions = PyArray_DIMS(ydot_ndarray),
        .data = PyArray_DATA(ydot_ndarray),
    };

    IVP_RHS_CALLBACK(t, &y, &y_dot);

    return PyLong_FromLong(0);
}

static PyMethodDef ivp_rhs_def = {
    "ivp_rhs_callback", c_to_py_wrapper, METH_VARARGS, NULL};

ImplInfo *load_backend(const char *impl_details,
                       size_t version_major,
                       size_t version_minor) {
    if (Py_IsInitialized()) {
        fprintf(stderr, "[backend_python] Backend is already initialized\n");
    } else {
        Py_Initialize();
    }

    // We need to `dlopen` the Python library, otherwise,
    // NumPy initialization fails.
    // Details:
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    char libpython_name[32];
    sprintf(libpython_name,
            "libpython%d.%d.so",
            PY_MAJOR_VERSION,
            PY_MINOR_VERSION);
    void *libpython = dlopen(libpython_name, RTLD_LAZY | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "[backend_python] Cannot open python library\n");
        exit(EXIT_FAILURE);
    }

    PyRun_SimpleString("import sys; "
                       "print('[backend_python]', sys.executable); "
                       "print('[backend_python]', sys.version)");

    import_array2("Failed to initialize NumPy C API", NULL);

    PyRun_SimpleString(
        "import numpy; "
        "print('[backend_python] NumPy version: ', numpy.__version__)");

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
    fprintf(
        stderr, "[backend_python] Provided module name: '%s'\n", moduleName);
    fprintf(stderr, "[backend_python] Provided class name: '%s'\n", className);
    pFileName = PyUnicode_FromString(moduleName);
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[backend_python] Failed to load module \"%s\"\n",
                moduleName);
        return NULL;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[backend_python] Failed to instantiate class %s\n",
                className);
        Py_DECREF(pClass);
        return NULL;
    }
    Py_DECREF(pClass);

    PythonImplInfo *impl_info = malloc(sizeof(PythonImplInfo));
    if (impl_info == NULL) {
        fprintf(stderr,
                "[dispatch_python] Could not allocate memory for Python "
                "implementation information\n");
    }
    impl_info->pInstance = pInstance;

    return (ImplInfo *)impl_info;
}

int run_interface_method(ImplInfo *impl_info,
                         const char *method,
                         OIFArgs *in_args,
                         OIFArgs *out_args) {
    if (impl_info->dh != OIF_LANG_PYTHON) {
        fprintf(stderr,
                "[dispatch_python] Provided implementation is not in Python\n");
        return -1;
    }
    PythonImplInfo *impl = (PythonImplInfo *)impl_info;

    PyObject *pFunc;
    PyObject *pArgs;
    PyObject *pValue;

    pFunc = PyObject_GetAttrString(impl->pInstance, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        int num_args = in_args->num_args + out_args->num_args;
        pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (int i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            } else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **)in_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, arr->data);
            } else if (in_args->arg_types[i] == OIF_CALLBACK) {
                OIFCallback *p = in_args->arg_values[i];
                if (p->src == OIF_LANG_PYTHON) {
                    pValue = (PyObject *)p->fn_p_py;
                } else if (p->src == OIF_LANG_C) {
                    fprintf(stderr,
                            "[dispatch_python] Check what callback to "
                            "wrap via src field\n");
                    IVP_RHS_CALLBACK = *(ivp_rhs_fp_t *)p->fn_p_c;
                    pValue = PyCFunction_New(&ivp_rhs_def, NULL);
                } else {
                    fprintf(
                        stderr,
                        "[dispatch_python] Cannot determine callback source\n");
                    pValue = NULL;
                }
                if (!PyCallable_Check(pValue)) {
                    fprintf(stderr,
                            "[dispatch_python] Input argument #%d "
                            "has type OIF_CALLBACK "
                            "but it is actually is not callable",
                            i);
                }
            } else {
                pValue = NULL;
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pFunc);
                fprintf(
                    stderr,
                    "[backend_python] Cannot convert input argument #%d with "
                    "provided type id %d\n",
                    i,
                    in_args->arg_types[i]);
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (int i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_INT) {
                pValue = PyLong_FromLong(*(int *)out_args->arg_values[i]);
            } else if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            } else if (out_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **)out_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(
                    arr->nd, arr->dimensions, NPY_FLOAT64, arr->data);
            } else {
                pValue = NULL;
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                fprintf(
                    stderr,
                    "[backend_python] Cannot convert out_arg %d of type %d\n",
                    i,
                    out_args->arg_types[i]);
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
        fprintf(
            stderr, "[dispatch_python] Cannot find function \"%s\"\n", method);
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

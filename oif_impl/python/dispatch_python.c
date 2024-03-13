#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <dlfcn.h>
#include <ffi.h>
#include <stdbool.h>

#include <stdlib.h>

#include <oif/api.h>
#include <oif/c_bindings.h>
#include <oif/dispatch_api.h>

typedef struct {
    ImplInfo base;
    PyObject *pInstance;
    PyObject *pCallbackClass;
} PythonImplInfo;

static int IMPL_COUNTER = 0;

static bool is_python_initialized_by_us = false;

PyObject *
instantiate_callback_class(void)
{
    char *moduleName = "_callback";
    char class_name[] = "PythonWrapperForCCallback";

    PyObject *pFileName = PyUnicode_FromString(moduleName);
    PyObject *pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Faile to load callback module\n");
        exit(1);
    }

    PyObject *CALLBACK_CLASS_P = PyObject_GetAttrString(pModule, class_name);
    if (CALLBACK_CLASS_P == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[backend_python] Cannot proceed as callback class %s could "
                "not be instantiated\n",
                class_name);
    }
    Py_INCREF(CALLBACK_CLASS_P);
    Py_DECREF(pModule);

    return CALLBACK_CLASS_P;
}

PyObject *
convert_oif_callback(OIFCallback *p)
{
    const char *id = "123";
    PyObject *fn_p = PyCapsule_New(p->fn_p_c, id, NULL);
    if (fn_p == NULL) {
        fprintf(stderr, "[dispatch_python] Could not create PyCapsule\n");
    }
    fprintf(stderr, "[dispatch_python] HARDCODE!!!!!!\n");
    unsigned int nargs = 3;
    PyObject *obj = Py_BuildValue("(N, I)", fn_p, nargs);
    if (obj == NULL) {
        fprintf(stderr, "[backend_python] Could not build arguments\n");
    }
    return obj;
}

ImplInfo *
load_backend(const char *impl_details, size_t version_major, size_t version_minor)
{
    if (Py_IsInitialized()) {
        fprintf(stderr, "[backend_python] Backend is already initialized\n");
    }
    else {
        Py_Initialize();
        if (IMPL_COUNTER == 0) {
            is_python_initialized_by_us = true;
        }
    }

    // We need to `dlopen` the Python library, otherwise,
    // NumPy initialization fails.
    // Details:
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
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
        }
        else {
            moduleName[i] = '\0';
            break;
        }
    }
    size_t offset = i + 1;
    for (; i < strlen(impl_details); ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            className[i - offset] = impl_details[i];
        }
        else {
            className[i] = '\0';
        }
    }

    PyObject *pFileName, *pModule;
    PyObject *pClass, *pInstance;
    PyObject *pInitArgs;
    fprintf(stderr, "[backend_python] Provided module name: '%s'\n", moduleName);
    fprintf(stderr, "[backend_python] Provided class name: '%s'\n", className);
    pFileName = PyUnicode_FromString(moduleName);
    if (pFileName == NULL) {
        fprintf(stderr,
                "[dispatch_python::load_impl] Provided moduleName '%s' "
                "could not be resolved to file name\n",
                moduleName);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Failed to load module \"%s\"\n", moduleName);
        return NULL;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        fprintf(stderr, "[backend_python] Failed to instantiate class %s\n", className);
        Py_DECREF(pClass);
        return NULL;
    }
    Py_INCREF(pInstance);
    Py_DECREF(pInitArgs);
    Py_DECREF(pClass);

    PythonImplInfo *impl_info = malloc(sizeof(*impl_info));
    if (impl_info == NULL) {
        fprintf(stderr,
                "[dispatch_python] Could not allocate memory for Python "
                "implementation information\n");
        return NULL;
    }
    impl_info->pInstance = pInstance;
    impl_info->pCallbackClass = NULL;

    IMPL_COUNTER++;

    return (ImplInfo *)impl_info;
}

int
run_interface_method(ImplInfo *impl_info, const char *method, OIFArgs *in_args,
                     OIFArgs *out_args)
{
    if (impl_info->dh != OIF_LANG_PYTHON) {
        fprintf(stderr, "[dispatch_python] Provided implementation is not in Python\n");
        return -1;
    }
    PythonImplInfo *impl = (PythonImplInfo *)impl_info;

    PyObject *pFunc;
    PyObject *pValue;

    pFunc = PyObject_GetAttrString(impl->pInstance, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        int num_args = in_args->num_args + out_args->num_args;
        PyObject *pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (int i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            }
            else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **)in_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64,
                                                   arr->data);
            }
            else if (in_args->arg_types[i] == OIF_CALLBACK) {
                OIFCallback *p = in_args->arg_values[i];
                if (p->src == OIF_LANG_PYTHON) {
                    pValue = (PyObject *)p->fn_p_py;
                    /*
                     * It is important to incref the callback pointed to
                     * with p->fn_p_py, because somehow a reference count
                     * to the ctypes object on Python side is not incremented.
                     * Therefore, when decref of `pArgs` occurs down below,
                     * the memory pointed to by p->fn_p_py is getting freed
                     * prematurely with the consequent segfault.
                     */
                    Py_INCREF(pValue);
                }
                else if (p->src == OIF_LANG_C) {
                    fprintf(stderr,
                            "[dispatch_python] Check what callback to "
                            "wrap via src field\n");
                    if (impl->pCallbackClass == NULL) {
                        impl->pCallbackClass = instantiate_callback_class();
                    }
                    PyObject *callback_args = convert_oif_callback(p);
                    pValue = PyObject_CallObject(impl->pCallbackClass, callback_args);
                    if (pValue == NULL) {
                        fprintf(stderr,
                                "[backend_python] Could not instantiate "
                                "Callback class for wrapping C functions\n");
                    }
                }
                else {
                    fprintf(stderr, "[dispatch_python] Cannot determine callback source\n");
                    pValue = NULL;
                }
                if (!PyCallable_Check(pValue)) {
                    fprintf(stderr,
                            "[dispatch_python] Input argument #%d "
                            "has type OIF_CALLBACK "
                            "but it is actually is not callable\n",
                            i);
                }
            }
            else {
                pValue = NULL;
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pFunc);
                fprintf(stderr,
                        "[backend_python] Cannot convert input argument #%d with "
                        "provided type id %d\n",
                        i, in_args->arg_types[i]);
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (int i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_INT) {
                pValue = PyLong_FromLong(*(int *)out_args->arg_values[i]);
            }
            else if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            }
            else if (out_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **)out_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64,
                                                   arr->data);
            }
            else {
                pValue = NULL;
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pFunc);
                fprintf(stderr, "[backend_python] Cannot convert out_arg %d of type %d\n", i,
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
        }
        else {
            Py_DECREF(pFunc);
            PyErr_Print();
            fprintf(stderr, "Call failed\n");
            return 2;
        }
    }
    else {
        if (PyErr_Occurred()) {
            fprintf(stderr, "[dispatch_python] ");
            PyErr_Print();
        }
        fprintf(stderr, "[dispatch_python] Cannot find function \"%s\"\n", method);
        Py_XDECREF(pFunc);
        return -1;
    }
    Py_DECREF(pFunc);

    return 0;
}

int
unload_impl(ImplInfo *impl_info_)
{
    if (impl_info_->dh != OIF_LANG_PYTHON) {
        fprintf(stderr,
                "[dispatch_python] unload_impl received non-Python "
                "implementation argument\n");
        return -1;
    }
    PythonImplInfo *impl_info = (PythonImplInfo *)impl_info_;

    Py_DECREF(impl_info->pInstance);
    Py_XDECREF(impl_info->pCallbackClass);
    IMPL_COUNTER--;

    if (is_python_initialized_by_us && (IMPL_COUNTER == 0)) {
        int status = Py_FinalizeEx();
        if (status < 0) {
            fprintf(stderr, "[dispatch_python] Py_FinalizeEx with status %d\n", status);
            return status;
        }
    }
    free(impl_info);

    return 0;
}

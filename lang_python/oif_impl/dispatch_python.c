#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <dlfcn.h>
#include <stdbool.h>

#include <stdlib.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include <oif/internal/bridge_api.h>

typedef struct {
    ImplInfo base;
    PyObject *pInstance;
    PyObject *pCallbackClass;
} PythonImplInfo;

static int IMPL_COUNTER = 0;

static bool is_python_initialized_by_us = false;

static char prefix[] = "dispatch_python";

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
        fprintf(stderr, "[%s] Failed to load callback module\n", prefix);
        exit(1);
    }

    PyObject *CALLBACK_CLASS_P = PyObject_GetAttrString(pModule, class_name);
    if (CALLBACK_CLASS_P == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[%s] Cannot proceed as callback class %s could "
                "not be instantiated\n",
                prefix, class_name);
    }
    Py_INCREF(CALLBACK_CLASS_P);
    Py_DECREF(pModule);

    return CALLBACK_CLASS_P;
}

static PyObject *
get_deserialization_function(void)
{
    char *moduleName = "_serialization";
    PyObject *pFileName = PyUnicode_FromString(moduleName);
    PyObject *pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to load `serialization` module\n", prefix);
        exit(1);
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "deserialize");
    if (pFunc == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Could not find function `deserialize`\n", prefix);
    }

    return pFunc;
}

static PyObject *
convert_oif_callback(OIFCallback *p)
{
    const char *id = "123";
    PyObject *fn_p = PyCapsule_New(p->fn_p_c, id, NULL);
    if (fn_p == NULL) {
        fprintf(stderr, "[%s] Could not create PyCapsule\n", prefix);
    }
    fprintf(stderr, "[%s] HARDCODE!!!!!!\n", prefix);
    unsigned int nargs = 4;
    PyObject *obj = Py_BuildValue("(N, I)", fn_p, nargs);
    if (obj == NULL) {
        fprintf(stderr, "[%s] Could not build arguments\n", prefix);
    }
    return obj;
}

ImplInfo *
load_impl(const char *impl_details, size_t version_major, size_t version_minor)
{
    PyObject *pFileName, *pModule;
    PyObject *pClass, *pInstance;
    PyObject *pFunc;
    PyObject *pInitArgs;
    PyObject *pArgs;
    PyObject *pValue;
    int status;

    (void)version_major;
    (void)version_minor;
    if (Py_IsInitialized()) {
        fprintf(stderr, "[%s] Backend is already initialized\n", prefix);
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
    char libpython_name[1024];
    pFileName = PyUnicode_DecodeFSDefault("sysconfig");
    if (pFileName == NULL) {
        fprintf(stderr, "[%s] Could not find `sysconfig` module file\n", prefix);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        fprintf(stderr, "[%s] Could not import `sysconfig` module\n", prefix);
        return NULL;
    }
    pFunc = PyObject_GetAttrString(pModule, "get_config_var");
    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        fprintf(stderr, "[%s] Could not find function `sysconfig.get_config_var`\n", prefix);
        return NULL;
    }
    pArgs = PyTuple_New(1);
    pValue = Py_BuildValue("s", "LIBDIR");
    status = PyTuple_SetItem(pArgs, 0, pValue);
    if (status != 0) {
        fprintf(stderr,
                "[%s] Could not build arguments for executing `sysconfig.get_config_var`\n",
                prefix);
        return NULL;
    }
    pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    if (pValue == NULL) {
        fprintf(stderr, "[%s] Could not execute `sysconfig.get_config_var`\n", prefix);
        return NULL;
    }
    const char *libpython_path = PyUnicode_AsUTF8(pValue);
    Py_DECREF(pValue);
    if (libpython_path == NULL) {
        fprintf(stderr, "[%s] Could not convert path to `libpython`\n", prefix);
        return NULL;
    }

    fprintf(stderr, "[%s] libpython path: %s\n", prefix, libpython_path);
    sprintf(libpython_name, "libpython%d.%d.so", PY_MAJOR_VERSION, PY_MINOR_VERSION);
    fprintf(stderr, "[%s] Loading %s\n", prefix, libpython_name);
    void *libpython = dlopen(libpython_name, RTLD_LAZY | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "[%s] Cannot open python library\n", prefix);
        exit(EXIT_FAILURE);
    }

    status = PyRun_SimpleString(
        "import sys; "
        "print('[dispatch_python]', sys.executable); "
        "print('[dispatch_python]', sys.version)");
    if (status < 0) {
        fprintf(stderr, "[%s] An error occurred when initializating Python\n", prefix);
        return NULL;
    }

    import_array2("Failed to initialize NumPy C API", NULL);

    status = PyRun_SimpleString(
        "import numpy; "
        "print('[dispatch_python] NumPy version: ', numpy.__version__)");
    if (status < 0) {
        fprintf(stderr, "[%s] An error occurred when initializating Python\n", prefix);
        return NULL;
    }

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

    fprintf(stderr, "[%s] Provided module name: '%s'\n", prefix, moduleName);
    fprintf(stderr, "[%s] Provided class name: '%s'\n", prefix, className);
    pFileName = PyUnicode_FromString(moduleName);
    if (pFileName == NULL) {
        fprintf(stderr,
                "[%s::load_impl] Provided moduleName '%s' "
                "could not be resolved to file name\n",
                prefix, moduleName);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to load module \"%s\"\n", prefix, moduleName);
        return NULL;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to instantiate class %s\n", prefix, className);
        Py_DECREF(pClass);
        return NULL;
    }
    Py_INCREF(pInstance);
    Py_DECREF(pInitArgs);
    Py_DECREF(pClass);

    PythonImplInfo *impl_info = malloc(sizeof(*impl_info));
    if (impl_info == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Python "
                "implementation information\n",
                prefix);
        return NULL;
    }
    impl_info->pInstance = pInstance;
    impl_info->pCallbackClass = NULL;

    IMPL_COUNTER++;

    return (ImplInfo *)impl_info;
}

int
call_impl(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    if (impl_info->dh != OIF_LANG_PYTHON) {
        fprintf(stderr, "[%s] Provided implementation is not in Python\n", prefix);
        return -1;
    }
    PythonImplInfo *impl = (PythonImplInfo *)impl_info;

    PyObject *pFunc;
    PyObject *pValue;

    pFunc = PyObject_GetAttrString(impl->pInstance, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        size_t num_args = in_args->num_args + out_args->num_args;
        PyObject *pArgs = PyTuple_New(num_args);

        // Convert input arguments.
        for (size_t i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            }
            else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
                OIFArrayF64 *arr = *(OIFArrayF64 **)in_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64,
                                                   arr->data);
            }
            else if (in_args->arg_types[i] == OIF_STR) {
                char *c_str = *((char **)in_args->arg_values[i]);
                pValue = PyUnicode_FromString(c_str);
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
                            "[%s] Check what callback to "
                            "wrap via src field\n",
                            prefix);
                    if (impl->pCallbackClass == NULL) {
                        impl->pCallbackClass = instantiate_callback_class();
                    }
                    PyObject *callback_args = convert_oif_callback(p);
                    pValue = PyObject_CallObject(impl->pCallbackClass, callback_args);
                    if (pValue == NULL) {
                        fprintf(stderr,
                                "[%s] Could not instantiate "
                                "Callback class for wrapping C functions\n",
                                prefix);
                    }
                }
                else {
                    fprintf(stderr, "[%s] Cannot determine callback source\n", prefix);
                    pValue = NULL;
                }
                if (!PyCallable_Check(pValue)) {
                    fprintf(stderr,
                            "[%s] Input argument #%zu "
                            "has type OIF_CALLBACK "
                            "but it is actually is not callable\n",
                            prefix, i);
                }
            }
            else if (in_args->arg_types[i] == OIF_USER_DATA) {
                OIFUserData *user_data = (OIFUserData *)in_args->arg_values[i];
                if (user_data->src == OIF_LANG_C) {
                    /* Treat the argument as a raw pointer. */
                    pValue = PyCapsule_New(user_data->c, NULL, NULL);
                }
                else if (user_data->src == OIF_LANG_PYTHON) {
                    pValue = user_data->py;
                }
                else {
                    fprintf(stderr, "[%s] Cannot handle user data with src %d\n", prefix,
                            user_data->src);
                    pValue = NULL;
                }
            }
            else if (in_args->arg_types[i] == OIF_CONFIG_DICT) {
                OIFConfigDict *dict = *((OIFConfigDict **)in_args->arg_values[i]);
                if (dict != NULL) {
                    PyObject *deserialize_fn = get_deserialization_function();
                    const uint8_t *buffer = oif_config_dict_get_serialized(dict);
                    size_t length = oif_config_dict_get_serialized_object_length(dict);
                    PyObject *serialized_dict = Py_BuildValue("y#", buffer, length);
                    PyObject *deserialize_args = Py_BuildValue("(O)", serialized_dict);
                    pValue = PyObject_CallObject(deserialize_fn, deserialize_args);
                    Py_DECREF(deserialize_args);
                    Py_DECREF(serialized_dict);
                }
                else {
                    pValue = PyDict_New();
                    if (pValue == NULL) {
                        fprintf(stderr, "[%s] Could not create a dictionary\n", prefix);
                    }
                }
            }
            else {
                pValue = NULL;
            }
            if (!pValue) {
                Py_DECREF(pArgs);
                Py_DECREF(pFunc);
                fprintf(stderr,
                        "[%s] Cannot convert input argument #%zu with "
                        "provided type id %d\n",
                        prefix, i, in_args->arg_types[i]);
                return 1;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (size_t i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_INT) {
                int *tmp = *(int **)out_args->arg_values[i];
                printf("tmp =====  %d\n", *tmp);
                pValue = PyArray_SimpleNewFromData(1, (intptr_t[1]){1}, NPY_INT32, tmp);
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
                fprintf(stderr, "[%s] Cannot convert out_arg %zu of type %d\n", prefix, i,
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
            fprintf(stderr, "[%s] Call failed\n", prefix);
            return 2;
        }
    }
    else {
        if (PyErr_Occurred()) {
            fprintf(stderr, "[%s] An error occurred during the call\n", prefix);
            PyErr_Print();
        }
        fprintf(stderr, "[%s] Cannot find function \"%s\"\n", prefix, method);
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
                "[%s] unload_impl received non-Python "
                "implementation argument\n",
                prefix);
        return -1;
    }
    PythonImplInfo *impl_info = (PythonImplInfo *)impl_info_;

    Py_DECREF(impl_info->pInstance);
    Py_XDECREF(impl_info->pCallbackClass);
    IMPL_COUNTER--;

    /*
     * We cannot finalize embedded Python at all as then it will be
     * a segmentation fault next time we try to initialize it and NumPy.
     * It is mentioned here:
     * https://cython.readthedocs.io/en/latest/src/tutorial/embedding.html
     */
    /* if (is_python_initialized_by_us && (IMPL_COUNTER == 0)) { */
    /*     int status = Py_FinalizeEx(); */
    /*     if (status < 0) { */
    /*         fprintf(stderr, "[%s] Py_FinalizeEx with status %d\n", prefix, status); */
    /*         return status; */
    /*     } */
    /* } */

    return 0;
}

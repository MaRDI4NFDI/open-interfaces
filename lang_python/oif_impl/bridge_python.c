#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <dlfcn.h>
#include <stdbool.h>

#include <stdio.h>
#include <stdlib.h>

#include <oif/api.h>
#include <oif/config_dict.h>
#include <oif/util.h>
#include <oif/internal/bridge_api.h>

typedef struct {
    ImplInfo base;
    PyObject *pInstance;
    PyObject *pCallbackClass;
} PythonImplInfo;

#ifdef __APPLE__
static char SHLIB_EXT[] = ".dylib";
#elif __linux__
static char SHLIB_EXT[] = ".so";
#endif

static int IMPL_COUNTER = 0;

static bool is_python_initialized_by_us = false;

static char prefix_[] = "bridge_python";

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
        fprintf(stderr, "[%s] Failed to load callback module\n", prefix_);
        exit(1);
    }

    PyObject *CALLBACK_CLASS_P = PyObject_GetAttrString(pModule, class_name);
    if (CALLBACK_CLASS_P == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[%s] Cannot proceed as callback class %s could "
                "not be instantiated\n",
                prefix_, class_name);
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
        fprintf(stderr, "[%s] Failed to load `serialization` module\n", prefix_);
        exit(1);
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "deserialize");
    if (pFunc == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Could not find function `deserialize`\n", prefix_);
    }

    return pFunc;
}

static PyObject *
convert_oif_callback(OIFCallback *p)
{
    const char *id = "123";
    PyObject *fn_p = PyCapsule_New(p->fn_p_c, id, NULL);

    if (fn_p == NULL) {
        fprintf(stderr, "[%s] Could not create PyCapsule\n", prefix_);
        return NULL;
    }

    fprintf(stderr, "[%s] HARDCODE!!!!!!\n", prefix_);
    unsigned int nargs = 4;
    PyObject *obj = Py_BuildValue("(N, I)", fn_p, nargs);
    if (obj == NULL) {
        fprintf(stderr, "[%s] Could not build arguments\n", prefix_);
    }
    return obj;
}

static inline PyObject *
get_numpy_array_from_oif_array_f64(OIFArrayF64 **value)
{
    PyObject *pValue = NULL;

    OIFArrayF64 *arr = *value;
    pValue = PyArray_SimpleNewFromData(arr->nd, arr->dimensions, NPY_FLOAT64, arr->data);
    if (pValue == NULL) {
        fprintf(stderr, "[%s] Could not create NumPy array\n", prefix_);
        return NULL;
    }

    if (arr->nd == 1) {
        PyArray_ENABLEFLAGS((PyArrayObject *)pValue,
                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    }
    else {
        if (OIF_ARRAY_C_CONTIGUOUS(arr)) {
            fprintf(stderr, "[%s] Got C-like array\n", prefix_);
            PyArray_CLEARFLAGS((PyArrayObject *)pValue, NPY_ARRAY_F_CONTIGUOUS);
            PyArray_ENABLEFLAGS((PyArrayObject *)pValue, NPY_ARRAY_C_CONTIGUOUS);
        }
        else if (OIF_ARRAY_F_CONTIGUOUS(arr)) {
            PyArray_CLEARFLAGS((PyArrayObject *)pValue, NPY_ARRAY_C_CONTIGUOUS);
            PyArray_ENABLEFLAGS((PyArrayObject *)pValue, NPY_ARRAY_F_CONTIGUOUS);

            // I am not quite sure why there is a duplication
            // of information in the PyArray_Object structure:
            // we not only preserve flags as NPY_ARRAY_F_CONTIGUOUS,
            // but also need to specify strides,
            // although the strides are completely deducable
            // from the dimensions, data type, and the order of the array.
            //
            // However, as we wrapped the existing array,
            // NumPy assumed that it is in the C style
            // and set the strides accordingly,
            // so we need to reverse them for the Fortran style.
            npy_intp *strides = PyArray_STRIDES((PyArrayObject *)pValue);
            for (int j = 0; j < arr->nd / 2; ++j) {
                npy_int tmp = strides[j];
                strides[j] = strides[arr->nd - j - 1];
                strides[arr->nd - j - 1] = tmp;
            }
        }
        else {
            fprintf(stderr, "[%s] Array is not C or Fortran contiguous. Cannot proceed\n",
                    prefix_);
            return NULL;
        }
    }

    return pValue;
}

static int
init_python_()
{
    int retval = -1;
    // For unknown reason, embedded Python does not take into the account
    // that we can use a virtual environment,
    // so we have to shape it a bit.
    const char *venv = getenv("VIRTUAL_ENV");
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    if (venv) {
        fprintf(stderr, "[%s] Detected virtual environment: %s\n", prefix_, venv);
        char buffer_prog_name[2048];
        sprintf(buffer_prog_name, "%s/bin/python", venv);

        status = PyConfig_SetBytesString(&config, &config.program_name, buffer_prog_name);
        if (PyStatus_Exception(status)) {
            logerr(prefix_, "init_python error: %s\n", status.err_msg);
            goto cleanup;
        }
    }
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        logerr(prefix_, "init_python error: %s\n", status.err_msg);
        goto cleanup;
    }
    if (IMPL_COUNTER == 0) {
        is_python_initialized_by_us = true;
    }

cleanup:
    PyConfig_Clear(&config);

    return 0;
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
        fprintf(stderr, "[%s] Backend is already initialized\n", prefix_);
    }
    else {
        int status;
        status = init_python_();
        if (status != 0) {
            return NULL;
        }
    }

    // We need to `dlopen` the Python library, otherwise,
    // NumPy initialization fails.
    // Details:
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    char libpython_name[1024];
    pFileName = PyUnicode_DecodeFSDefault("sysconfig");
    if (pFileName == NULL) {
        fprintf(stderr, "[%s] Could not find `sysconfig` module file\n", prefix_);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        fprintf(stderr, "[%s] Could not import `sysconfig` module\n", prefix_);
        return NULL;
    }
    pFunc = PyObject_GetAttrString(pModule, "get_config_var");
    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        fprintf(stderr, "[%s] Could not find function `sysconfig.get_config_var`\n", prefix_);
        return NULL;
    }
    pArgs = PyTuple_New(1);
    pValue = Py_BuildValue("s", "LIBDIR");
    status = PyTuple_SetItem(pArgs, 0, pValue);
    if (status != 0) {
        fprintf(stderr,
                "[%s] Could not build arguments for executing `sysconfig.get_config_var`\n",
                prefix_);
        return NULL;
    }
    pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    if (pValue == NULL) {
        fprintf(stderr, "[%s] Could not execute `sysconfig.get_config_var`\n", prefix_);
        return NULL;
    }
    const char *libpython_path = PyUnicode_AsUTF8(pValue);
    Py_DECREF(pValue);
    if (libpython_path == NULL) {
        fprintf(stderr, "[%s] Could not convert path to `libpython`\n", prefix_);
        return NULL;
    }

    fprintf(stderr, "[%s] libpython path: %s\n", prefix_, libpython_path);
    sprintf(libpython_name, "%s/libpython%d.%d%s", libpython_path, PY_MAJOR_VERSION,
            PY_MINOR_VERSION, SHLIB_EXT);
    fprintf(stderr, "[%s] Loading %s\n", prefix_, libpython_name);
    void *libpython = dlopen(libpython_name, RTLD_LAZY | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "[%s] Cannot open python library\n", prefix_);
        exit(EXIT_FAILURE);
    }

    status = PyRun_SimpleString(
        "import sys; "
        "print('[dispatch_python]', sys.executable); "
        "print('[dispatch_python]', sys.version)");
    if (status < 0) {
        fprintf(stderr, "[%s] An error occurred when initializating Python\n", prefix_);
        return NULL;
    }

    import_array2("Failed to initialize NumPy C API", NULL);

    status = PyRun_SimpleString(
        "import numpy; "
        "print('[dispatch_python] NumPy version: ', numpy.__version__)");
    if (status < 0) {
        fprintf(stderr, "[%s] An error occurred when initializating Python\n", prefix_);
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

    fprintf(stderr, "[%s] Provided module name: '%s'\n", prefix_, moduleName);
    fprintf(stderr, "[%s] Provided class name: '%s'\n", prefix_, className);
    pFileName = PyUnicode_FromString(moduleName);
    if (pFileName == NULL) {
        fprintf(stderr,
                "[%s::load_impl] Provided moduleName '%s' "
                "could not be resolved to file name\n",
                prefix_, moduleName);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to load module \"%s\"\n", prefix_, moduleName);
        return NULL;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    if (pClass == NULL || !PyCallable_Check(pClass)) {
        PyErr_Print();
        fprintf(stderr, "[%s] Cannot find class \"%s\"\n", prefix_, className);
        Py_DECREF(pModule);
        return NULL;
    }

    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to instantiate class %s\n", prefix_, className);
        Py_DECREF(pClass);
        return NULL;
    }
    Py_INCREF(pInstance);
    Py_DECREF(pInitArgs);
    Py_DECREF(pClass);

    PythonImplInfo *impl_info = oif_util_malloc(sizeof(*impl_info));
    if (impl_info == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Python "
                "implementation information\n",
                prefix_);
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
    int result = 1;
    if (impl_info->dh != OIF_LANG_PYTHON) {
        fprintf(stderr, "[%s] Provided implementation is not in Python\n", prefix_);
        return -1;
    }
    PythonImplInfo *impl = (PythonImplInfo *)impl_info;

    PyObject *pFunc;
    PyObject *pValue;

    size_t num_args = in_args->num_args + out_args->num_args;
    PyObject *pArgs = PyTuple_New(num_args);

    pFunc = PyObject_GetAttrString(impl->pInstance, method);

    if (pFunc && PyCallable_Check(pFunc)) {
        // Convert input arguments.
        for (size_t i = 0; i < in_args->num_args; ++i) {
            if (in_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
            }
            else if (in_args->arg_types[i] == OIF_ARRAY_F64) {
                pValue = get_numpy_array_from_oif_array_f64(in_args->arg_values[i]);
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
                else if (p->src == OIF_LANG_C || p->src == OIF_LANG_JULIA) {
                    fprintf(stderr,
                            "[%s] Check what callback to "
                            "wrap via src field\n",
                            prefix_);
                    if (impl->pCallbackClass == NULL) {
                        impl->pCallbackClass = instantiate_callback_class();
                    }
                    PyObject *callback_args = convert_oif_callback(p);
                    if (callback_args == NULL) {
                        fprintf(stderr,
                                "[%s] Could not convert OIFCallback to Python "
                                "callable object because function 'convert_oif_callback' "
                                "returned error\n",
                                prefix_);
                        pValue = NULL;
                        goto cleanup;
                    }
                    pValue = PyObject_CallObject(impl->pCallbackClass, callback_args);
                    if (pValue == NULL) {
                        fprintf(stderr,
                                "[%s] Could not instantiate "
                                "Callback class for wrapping C functions\n",
                                prefix_);
                    }
                }
                else {
                    fprintf(stderr, "[%s] Cannot determine callback source\n", prefix_);
                    pValue = NULL;
                }
                if (!PyCallable_Check(pValue)) {
                    fprintf(stderr,
                            "[%s] Input argument #%zu "
                            "has type OIF_CALLBACK "
                            "but it is actually is not callable\n",
                            prefix_, i);
                    goto cleanup;
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
                else if (user_data->src == OIF_LANG_JULIA) {
                    /* Treat the argument as a raw pointer. */
                    pValue = PyCapsule_New(user_data->jl, NULL, NULL);
                }
                else {
                    fprintf(stderr, "[%s] Cannot handle user data with src %d\n", prefix_,
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
                        fprintf(stderr, "[%s] Could not create a dictionary\n", prefix_);
                    }
                }
            }
            else {
                pValue = NULL;
            }

            if (!pValue) {
                fprintf(stderr,
                        "[%s] Cannot convert input argument #%zu with "
                        "provided type id %d\n",
                        prefix_, i, in_args->arg_types[i]);
                goto cleanup;
            }
            PyTuple_SetItem(pArgs, i, pValue);
        }
        // Convert output arguments.
        for (size_t i = 0; i < out_args->num_args; ++i) {
            if (out_args->arg_types[i] == OIF_INT) {
                int *tmp = *(int **)out_args->arg_values[i];
                pValue = PyArray_SimpleNewFromData(1, (intptr_t[1]){1}, NPY_INT32, tmp);
            }
            else if (out_args->arg_types[i] == OIF_FLOAT64) {
                pValue = PyFloat_FromDouble(*(double *)out_args->arg_values[i]);
            }
            else if (out_args->arg_types[i] == OIF_ARRAY_F64) {
                pValue = get_numpy_array_from_oif_array_f64(out_args->arg_values[i]);
            }
            else {
                pValue = NULL;
            }
            if (!pValue) {
                fprintf(stderr, "[%s] Cannot convert out_arg %zu of type %d\n", prefix_, i,
                        out_args->arg_types[i]);
                goto cleanup;
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
            fprintf(stderr, "[%s] Call failed\n", prefix_);
            return 2;
        }
    }
    else {
        if (PyErr_Occurred()) {
            fprintf(stderr, "[%s] An error occurred during the call\n", prefix_);
            PyErr_Print();
        }
        fprintf(stderr, "[%s] Cannot find function \"%s\"\n", prefix_, method);
        Py_XDECREF(pFunc);
        return -1;
    }
    Py_DECREF(pFunc);

    // Woo-hoo! We got a successful function call without errors.
    result = 0;

cleanup:
    Py_XDECREF(pValue);
    Py_XDECREF(pArgs);
    Py_XDECREF(pFunc);

    return result;
}

int
unload_impl(ImplInfo *impl_info_)
{
    if (impl_info_->dh != OIF_LANG_PYTHON) {
        fprintf(stderr,
                "[%s] unload_impl received non-Python "
                "implementation argument\n",
                prefix_);
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

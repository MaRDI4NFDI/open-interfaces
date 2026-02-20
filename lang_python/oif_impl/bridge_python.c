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
} PythonImplInfo;

#ifdef __APPLE__
static char SHLIB_EXT[] = ".dylib";
#elif __linux__
static char SHLIB_EXT[] = ".so";
#endif

static int IMPL_COUNTER = 0;

static bool is_python_initialized_by_us = false;

static void *LIBPYTHON_ = NULL;

static bool NUMPY_IS_INITIALIZED_ = false;
static bool EXTRA_MODULES_ARE_INITIALIZED_ = false;

static char prefix_[] = "bridge_python";

static PyObject *CALLBACK_CLASS_P_ = NULL;
static PyObject *DESERIALIZE_FN_ = NULL;

static int
instantiate_callback_class_(void)
{
    char const *const moduleName = "_callback";
    char const *const class_name = "PythonWrapperForCCallback";

    PyObject *pFileName = PyUnicode_FromString(moduleName);
    PyObject *pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to load callback module\n", prefix_);
        return 1;
    }

    CALLBACK_CLASS_P_ = PyObject_GetAttrString(pModule, class_name);
    Py_DECREF(pModule);

    if (CALLBACK_CLASS_P_ == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[%s] Cannot proceed as callback class %s could "
                "not be instantiated\n",
                prefix_, class_name);
        return 2;
    }

    return 0;
}

static int
instantiate_deserialization_function_(void)
{
    char const *const moduleName = "_serialization";
    PyObject *pFileName = PyUnicode_FromString(moduleName);
    PyObject *pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Failed to load `serialization` module\n", prefix_);
        return 1;
    }

    DESERIALIZE_FN_ = PyObject_GetAttrString(pModule, "deserialize");
    Py_DECREF(pModule);

    if (DESERIALIZE_FN_ == NULL) {
        PyErr_Print();
        fprintf(stderr, "[%s] Could not find function `deserialize`\n", prefix_);
        return 2;
    }

    return 0;
}

static PyObject *
build_callback_args_(OIFCallback *p)
{
    const char *const id = "123";
    PyObject *fn_p = PyCapsule_New(p->fn_p_c, id, NULL);

    if (fn_p == NULL) {
        fprintf(stderr, "[%s] Could not create PyCapsule\n", prefix_);
        return NULL;
    }

    PyObject *py_arg_types = PyList_New(p->nargs);
    if (py_arg_types == NULL) {
        logerr(prefix_, "Could not create a Python list for argument types\n");
        return NULL;
    }

    for (unsigned int i = 0; i < p->nargs; i++) {
        // PyList_SET_ITEM does not check if there is something at position i
        // which is completely safe here as we construct the list from scratch.
        PyList_SET_ITEM(py_arg_types, i, PyLong_FromLong(p->arg_types[i]));
    }

    PyObject *obj = Py_BuildValue("(N, I, N, I)", fn_p, p->nargs, py_arg_types, p->restype);
    if (obj == NULL) {
        PyErr_Print();
        Py_DECREF(fn_p);
        logerr(prefix_, "Could not build callback arguments\n");
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
                npy_intp tmp = strides[j];
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
    const char *const venv = getenv("VIRTUAL_ENV");
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

static void *
init_numpy_()
{
    PyObject *pFileName = NULL;
    PyObject *pModule = NULL;
    PyObject *pFunc = NULL;
    PyObject *pArgs = NULL;
    PyObject *pValue = NULL;

    int status;
    int nbytes_required;  // Number of required bytes in snprintf.
    int nbytes_written;   // Number of written bytes in snprintf.
    char *cmd = NULL;     // Command formatted in snprintf.
                          //
    // We need to `dlopen` the Python library, otherwise,
    // NumPy initialization fails.
    // Details:
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    char libpython_path[512];
    char libpython_name[1024];
    pFileName = PyUnicode_DecodeFSDefault("sysconfig");
    if (pFileName == NULL) {
        fprintf(stderr, "[%s] Could not find `sysconfig` module file\n", prefix_);
        return NULL;
    }
    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);
    pFileName = NULL;

    if (pModule == NULL) {
        fprintf(stderr, "[%s] Could not import `sysconfig` module\n", prefix_);
        return NULL;
    }
    pFunc = PyObject_GetAttrString(pModule, "get_config_var");
    Py_DECREF(pModule);
    pModule = NULL;

    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        fprintf(stderr, "[%s] Could not find function `sysconfig.get_config_var`\n", prefix_);
        goto cleanup;
    }

    pArgs = PyTuple_New(1);
    pValue = Py_BuildValue("s", "LIBDIR");
    status = PyTuple_SetItem(pArgs, 0, pValue);
    if (status != 0) {
        PyErr_Print();
        Py_DECREF(pValue);
        fprintf(stderr,
                "[%s] Could not build arguments for executing `sysconfig.get_config_var`\n",
                prefix_);
        goto cleanup;
    }

    PyObject *pResultValue = PyObject_CallObject(pFunc, pArgs);
    if (pResultValue == NULL) {
        fprintf(stderr, "[%s] Could not execute `sysconfig.get_config_var`\n", prefix_);
        goto cleanup;
    }

    nbytes_written =
        snprintf(libpython_path, sizeof libpython_path, "%s", PyUnicode_AsUTF8(pResultValue));
    Py_DECREF(pResultValue);
    if (nbytes_written < 0 || nbytes_written >= sizeof libpython_path) {
        logerr(prefix_, "Could not convert path to `libpython`");
        goto cleanup;
    }

    fprintf(stderr, "[%s] libpython path: %s\n", prefix_, libpython_path);
    nbytes_written = snprintf(libpython_name, sizeof libpython_name, "%s/libpython%d.%d%s",
                              libpython_path, PY_MAJOR_VERSION, PY_MINOR_VERSION, SHLIB_EXT);
    if (nbytes_written < 0 || nbytes_written >= sizeof libpython_name) {
        logerr(prefix_, "Could not build a filepath to the Python library");
        goto cleanup;
    }
    fprintf(stderr, "[%s] Loading %s\n", prefix_, libpython_name);
    LIBPYTHON_ = dlopen(libpython_name, RTLD_LAZY | RTLD_GLOBAL);
    if (LIBPYTHON_ == NULL) {
        logerr(prefix_, "Cannot open python library");
        goto cleanup;
    }

    const char cmd_sys_info[] =
        "import sys; "
        "print('[%s]', sys.executable); "
        "print('[%s]', sys.version)";
    nbytes_required = snprintf(NULL, 0, cmd_sys_info, prefix_, prefix_) + 1;
    cmd = malloc(sizeof(char) * nbytes_required);
    nbytes_written = snprintf(cmd, nbytes_required, cmd_sys_info, prefix_, prefix_);
    if (nbytes_written < 0 || nbytes_written >= nbytes_required) {
        logerr(prefix_,
               "An error occurred in formatting a string as a command for Python interpreter");
        goto cleanup;
    }
    status = PyRun_SimpleString(cmd);
    free(cmd);
    cmd = NULL;
    if (status < 0) {
        logerr(prefix_, "An error occurred when initializating Python");
        goto cleanup;
    }

    status = _import_array();
    if (status != 0) {
        PyErr_Print();
        logerr(prefix_, "Failed to initialize NumPy C API");
        goto cleanup;
    }

    const char *cmd_numpy_fmt =
        "import numpy; "
        "print('[%s] NumPy version: ', numpy.__version__)";
    nbytes_required = snprintf(NULL, 0, cmd_numpy_fmt, prefix_) + 1;
    cmd = malloc(sizeof(char) * nbytes_required);
    nbytes_written = snprintf(cmd, nbytes_required, cmd_numpy_fmt, prefix_);
    if (nbytes_written < 0 || nbytes_written >= nbytes_required) {
        logerr(prefix_,
               "An error occurred in formatting a string as a command for Python interpreter");
        goto cleanup;
    }
    status = PyRun_SimpleString(cmd);
    free(cmd);
    cmd = NULL;
    if (status < 0) {
        logerr(prefix_, "An error occurred when checking NumPy version");
        goto cleanup;
    }

    NUMPY_IS_INITIALIZED_ = true;

cleanup:
    Py_XDECREF(pArgs);
    Py_XDECREF(pFunc);

    if (cmd != NULL) {
        free(cmd);
    }
    return NULL;
}

static int
init_extra_(void)
{
    int status;

    status = instantiate_deserialization_function_();
    if (status) {
        logerr(prefix_, "Could not import the `deserialize` function");
        return 1;
    }

    status = instantiate_callback_class_();
    if (status) {
        logerr(prefix_, "Could not instantiate the Callback class");
        return 2;
    }

    EXTRA_MODULES_ARE_INITIALIZED_ = true;
    return 0;
}

static int
init_(void)
{
    int status;

    if (Py_IsInitialized()) {
        fprintf(stderr, "[%s] Backend is already initialized\n", prefix_);
    }
    else {
        status = init_python_();
        if (status != 0) {
            return 1;
        }
    }

    if (!NUMPY_IS_INITIALIZED_) {
        init_numpy_();
        if (!NUMPY_IS_INITIALIZED_) {
            return 2;
        }
    }

    if (!EXTRA_MODULES_ARE_INITIALIZED_) {
        status = init_extra_();
        if (status) {
            return 3;
        }
    }

    return 0;
}

static void
parse_module_and_class_names_(const char *impl_details, char *moduleName, char *className)
{
    // This function works only if `impl_details` do not have
    // extra whitespace, otherwise there is discrepancy
    // between the indices in the written strings and `impl_details`.
    // TODO: Make the function more robust in that regard.
    const size_t N = strlen(impl_details) + 1;
    size_t i;
    for (i = 0; i < N; ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            moduleName[i] = impl_details[i];
        }
        else {
            moduleName[i] = '\0';
            break;
        }
    }
    size_t offset = i + 1;
    for (; i < N; ++i) {
        if (impl_details[i] != ' ' && impl_details[i] != '\0') {
            className[i - offset] = impl_details[i];
        }
        else {
            className[i] = '\0';
        }
    }
}

ImplInfo *
load_impl(const char *impl_details, size_t version_major, size_t version_minor)
{
    PyObject *pFileName = NULL;
    PyObject *pModule = NULL;
    PyObject *pClass = NULL;
    PyObject *pInitArgs = NULL;
    PyObject *pInstance = NULL;

    ImplInfo *result = NULL;
    int status;

    (void)version_major;
    (void)version_minor;

    status = init_();
    if (status) {
        logerr(prefix_, "Could not initialize important things. Cannot proceed");
        return NULL;
    }

    char moduleName[512] = "\0";
    char className[512] = "\0";
    parse_module_and_class_names_(impl_details, moduleName, className);

    fprintf(stderr, "[%s] Provided module name: '%s'\n", prefix_, moduleName);
    fprintf(stderr, "[%s] Provided class name: '%s'\n", prefix_, className);

    pFileName = PyUnicode_FromString(moduleName);
    if (pFileName == NULL) {
        PyErr_Print();
        fprintf(stderr,
                "[%s::load_impl] Provided moduleName '%s' "
                "could not be resolved to file name\n",
                prefix_, moduleName);
        goto cleanup;
    }
    pModule = PyImport_Import(pFileName);

    if (pModule == NULL) {
        PyErr_Print();
        logerr(prefix_, "Failed to load module \"%s\"\n", moduleName);
        goto cleanup;
    }

    pClass = PyObject_GetAttrString(pModule, className);
    if (pClass == NULL || !PyCallable_Check(pClass)) {
        PyErr_Print();
        logerr(prefix_, "Cannot find class \"%s\"\n", className);
        goto cleanup;
    }

    pInitArgs = Py_BuildValue("()");
    pInstance = PyObject_CallObject(pClass, pInitArgs);
    if (pInstance == NULL) {
        PyErr_Print();
        logerr(prefix_, "Failed to instantiate class %s\n", className);
        goto cleanup;
    }

    PythonImplInfo *impl_info = oif_util_malloc(sizeof(*impl_info));
    if (impl_info == NULL) {
        fprintf(stderr,
                "[%s] Could not allocate memory for Python "
                "implementation information\n",
                prefix_);
        Py_DECREF(pInstance);
        goto cleanup;
    }
    impl_info->pInstance = pInstance;

    IMPL_COUNTER++;

    result = (ImplInfo *)impl_info;

cleanup:
    Py_XDECREF(pFileName);
    Py_XDECREF(pModule);
    Py_XDECREF(pClass);
    Py_XDECREF(pInitArgs);
    /* Py_XDECREF(pInstance);  We keep it inside an impl_info object. */

    return result;
}

int
call_impl(ImplInfo *impl_info, const char *method, OIFArgs *in_args, OIFArgs *out_args,
          OIFArgs *return_args)
{
    int result = 1;
    if (impl_info->dh != OIF_LANG_PYTHON) {
        fprintf(stderr, "[%s] Provided implementation is not in Python\n", prefix_);
        return -1;
    }
    PythonImplInfo *impl = (PythonImplInfo *)impl_info;

    int status;  // To check codes of functions that return 0 on success.

    PyObject *pFunc = NULL;
    PyObject *pValue = NULL;

    size_t num_args = in_args->num_args + out_args->num_args;
    PyObject *pArgs = PyTuple_New(num_args);
    if (pArgs == NULL) {
        logerr(prefix_, "Could not create tuple for function arguments of size %zu", num_args);
        goto cleanup;
    }

    pFunc = PyObject_GetAttrString(impl->pInstance, method);

    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        if (PyErr_Occurred()) {
            fprintf(stderr, "[%s] An error occurred during the call\n", prefix_);
            PyErr_Print();
        }
        fprintf(stderr, "[%s] Cannot find function \"%s\"\n", prefix_, method);
        goto cleanup;
    }

    // Convert input arguments.
    for (size_t i = 0; i < in_args->num_args; ++i) {
        if (in_args->arg_types[i] == OIF_TYPE_F64) {
            pValue = PyFloat_FromDouble(*(double *)in_args->arg_values[i]);
        }
        else if (in_args->arg_types[i] == OIF_TYPE_ARRAY_F64) {
            pValue = get_numpy_array_from_oif_array_f64(in_args->arg_values[i]);
        }
        else if (in_args->arg_types[i] == OIF_TYPE_STRING) {
            char *c_str = *((char **)in_args->arg_values[i]);
            pValue = PyUnicode_FromString(c_str);
        }
        else if (in_args->arg_types[i] == OIF_TYPE_CALLBACK) {
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
                PyObject *callback_args = build_callback_args_(p);
                if (callback_args == NULL) {
                    logerr(prefix_,
                           "Could not convert OIFCallback to Python "
                           "callable object returned error\n");
                    goto cleanup;
                }

                pValue = PyObject_CallObject(CALLBACK_CLASS_P_, callback_args);
                Py_DECREF(callback_args);
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
                        "has type OIF_TYPE_CALLBACK "
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
                Py_INCREF(pValue);
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
        else if (in_args->arg_types[i] == OIF_TYPE_CONFIG_DICT) {
            OIFConfigDict *dict = *((OIFConfigDict **)in_args->arg_values[i]);
            if (dict != NULL) {
                const uint8_t *buffer = oif_config_dict_get_serialized(dict);
                size_t length = oif_config_dict_get_serialized_object_length(dict);
                PyObject *serialized_dict = Py_BuildValue("y#", buffer, length);
                PyObject *deserialize_args = Py_BuildValue("(O)", serialized_dict);
                pValue = PyObject_CallObject(DESERIALIZE_FN_, deserialize_args);
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
        status = PyTuple_SetItem(pArgs, i, pValue);
        if (status) {
            PyErr_Print();
            Py_DECREF(pValue);
            logerr(prefix_, "Could not build a tuple of arguments");
            goto cleanup;
        }
    }

    // Convert output arguments.
    for (size_t i = 0; i < out_args->num_args; ++i) {
        if (out_args->arg_types[i] == OIF_TYPE_ARRAY_F64) {
            pValue = get_numpy_array_from_oif_array_f64(out_args->arg_values[i]);
        }
        else {
            pValue = NULL;
        }
        if (pValue == NULL) {
            fprintf(stderr, "[%s] Cannot convert out_arg %zu of type %d\n", prefix_, i,
                    out_args->arg_types[i]);
            goto cleanup;
        }
        status = PyTuple_SetItem(pArgs, i + in_args->num_args, pValue);
        if (status) {
            PyErr_Print();
            Py_DECREF(pValue);
            logerr(prefix_, "Could not build a tuple of arguments");
            goto cleanup;
        }
    }

    // Invoke function.
    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue == NULL) {
        PyErr_Print();
        logerr(prefix_, "Call to the method '%s' has failed\n", method);
        goto cleanup;
    }

    // Convert return arguments.
    if (return_args != NULL) {
        if (return_args->num_args > 1 && !PyTuple_Check(pValue)) {
            logerr(prefix_,
                   "Expected %d return arguments, however the method '%s' "
                   "has not returned a tuple",
                   return_args->num_args, method);
            goto cleanup;
        }
        if (return_args->num_args != 1 && return_args->num_args != PyTuple_Size(pValue)) {
            logerr(prefix_,
                   "Expected %d returned arguments, "
                   "got %d instead in the call to the method '%s'",
                   return_args->num_args, PyTuple_Size(pValue), method);
            goto cleanup;
        }

        for (size_t i = 0; i < return_args->num_args; ++i) {
            PyObject *val = PyTuple_GetItem(pValue, i);
            switch (return_args->arg_types[i]) {
                case OIF_TYPE_I32:
                    if (!PyLong_Check(val)) {
                        logerr(prefix_, "Expected Python integer object, but did not get it");
                        goto cleanup;
                    }

                    long tmp = PyLong_AsLong(val);
                    if (tmp == -1 && PyErr_Occurred()) {
                        logerr(prefix_, "Could not convert Python object to C long value");
                        goto cleanup;
                    }
                    if (tmp >= INT32_MIN && tmp <= INT32_MAX) {
                        return_args->arg_values[i] = oif_util_malloc(sizeof(int32_t));
                        *(int32_t *)return_args->arg_values[i] = tmp;
                    }
                    else {
                        logerr(prefix_, "Return value is outside of the range of int32");
                        goto cleanup;
                    }
                    break;
                case OIF_TYPE_STRING:
                    if (!PyUnicode_Check(val)) {
                        logerr(prefix_, "Expected Unicode string object, but did not get it");
                        goto cleanup;
                    }
                    // Quote from docs for `PyUnicode_AsUTF8`:
                    // The caller is not responsible for deallocating the buffer.
                    const char *s = PyUnicode_AsUTF8(val);
                    if (s == NULL) {
                        logerr(prefix_, "Expected Unicode string object, but did not get it");
                    }
                    size_t length = PyUnicode_GET_LENGTH(val) + 1;
                    return_args->arg_values[i] = oif_util_malloc(sizeof(char) * length);
                    snprintf(return_args->arg_values[i], length, "%s", s);
                    break;
                default:
                    logerr(prefix_, "Return type with id '%d' is not supported yet",
                           return_args->arg_types[i]);
                    goto cleanup;
            }
        }
    }

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

#if defined(__GNUC__)
void __attribute__((destructor))
dtor()
{
    Py_XDECREF(CALLBACK_CLASS_P_);
    Py_XDECREF(DESERIALIZE_FN_);

    dlclose(LIBPYTHON_);
}
#endif

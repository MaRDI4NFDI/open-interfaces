// Dispatch library that is called from other languages, and dispatches it
// to the appropriate backend.
#include "dispatch.h"

#include <dlfcn.h>
#include <ffi.h>
#include <stdio.h>
#include <string.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

enum
{
    BACKEND_C,
    BACKEND_CXX,
    BACKEND_PYTHON,
    BACKEND_JULIA,
    BACKEND_R,
};

char OIF_BACKEND_C_SO[] = "./liboif_backend_c.so";

BackendHandle load_backend(
    const char *backend_name,
    const char *operation,
    size_t version_major,
    size_t version_minor)
{
    BackendHandle bh;
    if (strcmp(backend_name, "c") == 0)
    {
        bh = load_backend_c(operation, version_major, version_minor);
    }
    else if (strcmp(backend_name, "python") == 0)
    {
        bh = load_backend_python(operation, version_major, version_minor);
    }
    else
    {
        fprintf(stderr, "[dispatch] Cannot load backend: %s\n", backend_name);
        exit(EXIT_FAILURE);
    }

    return bh;
}

BackendHandle load_backend_c(
    const char *operation,
    size_t version_major,
    size_t version_minor)
{
    return BACKEND_C;
}

BackendHandle load_backend_python(
    const char *operation,
    size_t version_major,
    size_t version_minor)
{
    // Start Python interpreter here.
    fprintf(stderr, "[dispatch] This is not yet implemented correctly\n");
    exit(EXIT_FAILURE);
    return BACKEND_PYTHON;
}

int call_interface_method(
    BackendHandle bh,
    const char *method,
    OIFArgs *args,
    OIFArgs *retvals)
{
    int status;
    switch (bh)
    {
    case BACKEND_C:
        status = run_interface_method_c(method, args, retvals);
        break;
    case BACKEND_PYTHON:
        status = run_interface_method_python(method, args, retvals);
        break;
    default:
        fprintf(stderr, "[dispatch] Cannot call interface on backend handle: '%zu'", bh);
        exit(EXIT_FAILURE);
    }
    return status;
}

int run_interface_method_c(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    void *lib_handle = dlopen(OIF_BACKEND_C_SO, RTLD_LOCAL | RTLD_LAZY);
    if (lib_handle == NULL)
    {
        fprintf(stderr, "[dispatch] Cannot load shared library %s\n", OIF_BACKEND_C_SO);
        exit(EXIT_FAILURE);
    }
    void *func = dlsym(lib_handle, method);
    if (func == NULL)
    {
        fprintf(stderr, "[dispatch] Cannot load interface '%s'\n", method);
        exit(EXIT_FAILURE);
    }

    int num_in_args = in_args->num_args;
    int num_out_args = out_args->num_args;
    int num_total_args = num_in_args + num_out_args;

    ffi_cif cif;
    ffi_type **arg_types = malloc(num_total_args * sizeof(ffi_type));
    void **arg_values = malloc(num_total_args * sizeof(void *));

    // Merge input and output argument types together in `arg_types` array.
    for (size_t i = 0; i < num_in_args; ++i)
    {
        if (in_args->arg_types[i] == OIF_FLOAT64)
        {
            arg_types[i] = &ffi_type_double;
        }
        else if (in_args->arg_types[i] == OIF_FLOAT64_P)
        {
            arg_types[i] = &ffi_type_pointer;
        }
        else
        {
            fflush(stdout);
            fprintf(stderr, "[dispatch] Unknown input arg type: %d\n", in_args->arg_types[i]);
            exit(EXIT_FAILURE);
        }
    }
    for (size_t i = num_in_args; i < num_total_args; ++i)
    {
        printf("Processing out_args[%zu] = %u\n", i - num_in_args, out_args->arg_types[i - num_in_args]);
        if (out_args->arg_types[i - num_in_args] == OIF_FLOAT64)
        {
            arg_types[i] = &ffi_type_double;
        }
        else if (out_args->arg_types[i - num_in_args] == OIF_FLOAT64_P)
        {
            arg_types[i] = &ffi_type_pointer;
        }
        else
        {
            fflush(stdout);
            fprintf(stderr, "[dispatch] Unknown output arg type: %d\n", out_args->arg_types[i - num_in_args]);
            exit(EXIT_FAILURE);
        }
    }

    ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, num_total_args, &ffi_type_uint, arg_types);
    if (status == FFI_OK)
    {
        printf("[backend_c] ffi_prep_cif returned FFI_OK\n");
    }
    else
    {
        fflush(stdout);
        fprintf(stderr, "[dispatch] ffi_prep_cif was not OK");
        exit(EXIT_FAILURE);
    }

    // Merge input and output argument values together in `arg_values` array.
    for (size_t i = 0; i < num_in_args; ++i)
    {
        arg_values[i] = in_args->arg_values[i];
    }
    for (size_t i = num_in_args; i < num_total_args; ++i)
    {
        arg_values[i] = out_args->arg_values[i - num_in_args];
    }

    unsigned result;
    ffi_call(&cif, FFI_FN(func), &result, arg_values);

    printf("Result is %u\n", result);
    fflush(stdout);

    return 0;
}

int run_interface_method_python(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    PyObject *pFileName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    Py_Initialize();
    printf("Provided module name: %s\n", method);
    pFileName = PyUnicode_DecodeFSDefault();
    printf("PyUnicode_DecodeFSDefault module name: %s\n", PyUnicode_AsUTF8(pFileName));

    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    pFunc = PyObject_GetAttrString(pModule, argv[2]);
    /* pFunc is a new reference */

    if (pFunc && PyCallable_Check(pFunc))
    {
        pArgs = PyTuple_New(argc - 3);
        for (int i = 3; i < argc; ++i)
        {
            pValue = PyLong_FromLong(atoi(argv[i]));
            if (!pValue)
            {
                Py_DECREF(pArgs);
                Py_DECREF(pModule);
                fprintf(stderr, "Cannot convert argument\n");
                return 1;
            }
            /* pValue reference is stolen here */
            PyTuple_SetItem(pArgs, i - 3, pValue);
        }
        pValue = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs);
        if (pValue != NULL)
        {
            printf("Result of call: %ld\n", PyLong_AsLong(pValue));
            Py_DECREF(pValue);
        }
        else
        {
            Py_DECREF(pFunc);
            Py_DECREF(pModule);
            PyErr_Print();
            fprintf(stderr, "Call failed\n");
            return EXIT_FAILURE;
        }
    }
    else
    {
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);

    if (Py_FinalizeEx() < 0)
    {
        return 120;
    }

    return 0;
}

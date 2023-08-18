#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <ffi.h>

#include "dispatch.h"


int run_interface_method_python(const char *method, OIFArgs *in_args, OIFArgs *out_args)
{
    PyObject *pFileName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    Py_Initialize();
    printf("Provided module name: %s\n", method);
    pFileName = PyUnicode_DecodeFSDefault(method);
    printf("PyUnicode_DecodeFSDefault module name: %s\n", PyUnicode_AsUTF8(pFileName));

    pModule = PyImport_Import(pFileName);
    Py_DECREF(pFileName);

    if (pModule == NULL)
    {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", method);
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

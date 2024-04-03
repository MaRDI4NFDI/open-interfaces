#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <ffi.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "oif/api.h"

typedef struct {
    PyObject_HEAD void *fn_p;   // raw C function pointer retrieved from PyCapsule
    unsigned int nargs;         // Number of function arguments
    OIFArgType *oif_arg_types;  // Arg types used in OpenInterFaces
    ffi_cif *cif_p;             // Pointer to libffi context object
    ffi_type **arg_types;       // Arg types in terms of libffi
    void **arg_values;          // Memory for the data converted to C
    unsigned int narray_args;   // Number of arrays among argument types
    OIFArrayF64 **oif_arrays;
} PythonWrapperForCCallbackObject;

static void
PythonWrapperForCCallback_dealloc(PythonWrapperForCCallbackObject *self)
{
    unsigned int nargs = self->nargs;
    unsigned int narray_args = self->narray_args;
    for (unsigned int i = 0; i < narray_args; ++i) {
        if (self->oif_arrays[i] != NULL) {
            free(self->oif_arrays[i]);
        }
    }
    free(self->oif_arrays);

    for (unsigned int i = 0; i < nargs; ++i) {
        if (self->arg_values[i] != NULL) {
            free(self->arg_values[i]);
        }
    }
    free(self->arg_values);

    free(self->arg_types);
    free(self->cif_p);
    free(self->oif_arg_types);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
PythonWrapperForCCallback_init(PythonWrapperForCCallbackObject *self, PyObject *args,
                               PyObject *Py_UNUSED(kwds))
{
    PyObject *capsule;
    unsigned int nargs;

    /* These lines must also parse for argument types list */
    // O = object, I = unsigned int
    if (!PyArg_ParseTuple(args, "OI", &capsule, &nargs)) {
        fprintf(stderr, "[_callback] Could not parse arguments\n");
        return -1;
    }

    self->fn_p = PyCapsule_GetPointer(capsule, "123");

    assert(nargs == 4);
    nargs = 4;
    self->nargs = nargs;

    self->oif_arg_types = malloc(sizeof(OIFArgType) * nargs);
    if (self->oif_arg_types == NULL) {
        fprintf(stderr, "[_callback] Could not allocated memory for oif_arg_types\n");
        goto fail_clean_self;
    }
    self->oif_arg_types[0] = OIF_FLOAT64;
    self->oif_arg_types[1] = OIF_ARRAY_F64;
    self->oif_arg_types[2] = OIF_ARRAY_F64;
    self->oif_arg_types[3] = OIF_USER_DATA;

    self->cif_p = malloc(sizeof(ffi_cif));
    if (self->cif_p == NULL) {
        goto fail_clean_oif_arg_types;
    }

    self->arg_types = malloc(nargs * sizeof(ffi_type *));
    if (self->arg_types == NULL) {
        fprintf(stderr, "[_callback] Could not allocate memory for `arg_types`\n");
        goto fail_clean_cif_p;
    }

    // We need to allocate memory for all values, to make sure
    // that the lifetime of the arguments ends after `ffi_call` below.
    self->arg_values = calloc(nargs, sizeof(void *));
    if (self->arg_values == NULL) {
        fprintf(stderr, "[_callback] Could not allocate memory for `arg_values`\n");
        goto fail_clean_arg_types;
    }
    unsigned int narray_args = 0;  // Number of arguments of type `OIFArrayF64 *`.
    for (size_t i = 0; i < nargs; ++i) {
        if (self->oif_arg_types[i] == OIF_INT) {
            fprintf(stderr,
                    "[_callback] WARNING: There must be better support "
                    "for integer types\n");
            self->arg_types[i] = &ffi_type_sint64;
            self->arg_values[i] = malloc(sizeof(int));
        }
        else if (self->oif_arg_types[i] == OIF_FLOAT64) {
            self->arg_types[i] = &ffi_type_double;
            self->arg_values[i] = malloc(sizeof(double));
        }
        else if (self->oif_arg_types[i] == OIF_ARRAY_F64) {
            self->arg_types[i] = &ffi_type_pointer;
            self->arg_values[i] = malloc(sizeof(OIFArrayF64 **));
            narray_args++;
        }
        else if (self->oif_arg_types[i] == OIF_USER_DATA) {
            self->arg_types[i] = &ffi_type_pointer;
            self->arg_values[i] = malloc(sizeof(void *));
        }
        else {
            fprintf(stderr, "[_callback] Unknown input arg type: %d\n",
                    self->oif_arg_types[i]);
            goto fail_clean_arg_values;
        }
        if (self->arg_values[i] == NULL) {
            fprintf(stderr, "[_callback] Could not allocate memory for element #%zu\n", i);
            goto fail_clean_arg_values;
        }
    }

    self->narray_args = narray_args;

    self->oif_arrays = calloc(narray_args, sizeof(OIFArrayF64 *));
    if (self->oif_arrays == NULL) {
        fprintf(stderr, "[_callback] Could not allocate memory for `oif_arrays`\n");
        goto fail_clean_arg_values;
    }
    for (Py_ssize_t i = 0; i < narray_args; ++i) {
        self->oif_arrays[i] = malloc(sizeof(OIFArrayF64));
        if (self->oif_arrays[i] == NULL) {
            fprintf(stderr, "[_callback] Could not allocate memory for `oif_arrays[%ld]`\n",
                    i);
            goto fail_clean_oif_arrays;
        }
    }

    return 0;

fail_clean_oif_arrays:
    for (unsigned int i = 0; i < narray_args; ++i) {
        if (self->oif_arrays[i] != NULL) {
            free(self->oif_arrays[i]);
        }
    }
    free(self->oif_arrays);
fail_clean_arg_values:
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (self->arg_values[i] != NULL) {
            free(self->arg_values[i]);
        }
    }
    free(self->arg_values);
fail_clean_arg_types:
    free(self->arg_types);
fail_clean_cif_p:
    free(self->cif_p);
fail_clean_oif_arg_types:
    free(self->oif_arg_types);
fail_clean_self:
    Py_DECREF(self);

    return -1;
}

static PyObject *
PythonWrapperForCCallback_call(PyObject *myself, PyObject *args, PyObject *Py_UNUSED(kwds))
{
    PythonWrapperForCCallbackObject *self = (PythonWrapperForCCallbackObject *)myself;
    PyObject *retval = NULL;

    /* if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &py_args)) { */
    /*     fprintf(stderr, "[_callback] Could not parse function arguments\n");
     */
    /*     return NULL; */
    /* } */
    PyObject *py_args = args;

    Py_ssize_t nargs_s = PyTuple_Size(py_args);
    if (nargs_s < 0) {
        fprintf(stderr,
                "[_callback] Unexpected negative value for the size "
                "of the tuple of args for the callback function\n");
        return NULL;
    }
    unsigned int nargs;
    if (nargs_s <= UINT_MAX) {
        nargs = (unsigned int)nargs_s;  // Explicit cast to eliminate compiler warning
    }
    else {
        fprintf(stderr,
                "[_callback] Could not convert size of the tuple of args "
                "to 'unsigned int' type\n");
        return NULL;
    }
    // assert(nargs == self->nargs);

    OIFArgType *arg_type_ids = self->oif_arg_types;

    ffi_cif cif;
    void **arg_values = self->arg_values;
    OIFArrayF64 **oif_arrays = self->oif_arrays;

    void *fn_p = self->fn_p;

    // Prepare function arguments for FFI expectations (pointers)
    // and convert NumPy arrays to OIFArrayF64 structs.
    for (size_t i = 0, j = 0; i < nargs; ++i) {
        PyObject *arg = PyTuple_GetItem(py_args, i);
        if (arg_type_ids[i] == OIF_FLOAT64) {
            if (!PyFloat_Check(arg)) {
                fprintf(stderr, "[_callback] Expected PyFloat object.\n");
                return NULL;
            }
            double *double_value = arg_values[i];
            *double_value = PyFloat_AsDouble(arg);
        }
        else if (arg_type_ids[i] == OIF_ARRAY_F64) {
            PyArrayObject *py_arr = (PyArrayObject *)arg;
            if (!PyArray_Check(py_arr)) {
                fprintf(stderr,
                        "[_callback] Expected PyArrayObject (NumPy ndarray) "
                        "object\n");
                return NULL;
            }
            oif_arrays[j]->nd = PyArray_NDIM(py_arr);
            oif_arrays[j]->dimensions = PyArray_DIMS(py_arr);
            oif_arrays[j]->data = PyArray_DATA(py_arr);
            // We always pass array data structure as pointer: `OIFArrayF64 *`,
            // and FFI requires pointer to function arguments;
            // hence, we need to obtain `OIFArrayF64 **`.
            OIFArrayF64 **pp = arg_values[i];
            *pp = oif_arrays[j];
            j++;
        }
        else if (arg_type_ids[i] == OIF_USER_DATA) {
            void **p_user_data = arg_values[i];
            *p_user_data = PyCapsule_GetPointer(arg, NULL);
        }
        else {
            fprintf(stderr, "[_callback] Unknown input arg type: %d\n", arg_type_ids[i]);
            return NULL;
        }
    }

    ffi_status status =
        ffi_prep_cif(&cif, FFI_DEFAULT_ABI, nargs, &ffi_type_sint, self->arg_types);
    if (status != FFI_OK) {
        fflush(stdout);
        fprintf(stderr, "[_callback] ffi_prep_cif was not OK");
        return NULL;
    }

    int result;
    ffi_call(&cif, FFI_FN(fn_p), &result, arg_values);

    retval = PyLong_FromLong(result);

    return retval;
}

static PyMemberDef PythonWrapperForCCallback_members[] = {
    {NULL} /* Sentinel */
};

static PyTypeObject PythonWrapperForCCallbackType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "callback.PythonWrapperForCCallback",
    .tp_doc = PyDoc_STR("Python wrapper for a C callback function"),
    .tp_basicsize = sizeof(PythonWrapperForCCallbackObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PythonWrapperForCCallback_init,
    .tp_dealloc = (destructor)PythonWrapperForCCallback_dealloc,
    .tp_call = (ternaryfunc)PythonWrapperForCCallback_call,
    .tp_members = PythonWrapperForCCallback_members,
    // .tp_methods = PythonWrapperForCCallback_methods,
};

static PyObject *
make_pycapsule(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *py_fn_p;
    void *fn_p;
    PyObject *capsule;

    if (!PyArg_ParseTuple(args, "O", &py_fn_p)) {
        fprintf(stderr, "[_callback] Could not parse make_capsule arguments\n");
        return NULL;
    }
    fn_p = PyLong_AsVoidPtr(py_fn_p);
    capsule = PyCapsule_New(fn_p, "123", NULL);
    if (capsule == NULL) {
        fprintf(stderr, "[_callback] Could not create a capsule\n");
        return NULL;
    }

    return capsule;
}

static PyMethodDef callback_methods[] = {
    {"make_pycapsule", make_pycapsule, METH_VARARGS,
     "Make a PyCapsule for C function pointer"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyDoc_STRVAR(callback_doc,
             "This module contains a function to invoke a given C function from Python "
             "converting NumPy arrays to their OIFArrayF64 analogs.");

static struct PyModuleDef callbackmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "callback",  /* name of module */
    .m_doc = callback_doc, /* module documentation, may be NULL */
    .m_size = -1,          /* size of per-interpreter state of the module,
                              or -1 if the module keeps state in global variables. */
    .m_methods = callback_methods,
};

static void
test_fn_(void)
{
    fprintf(stdout, "[_callback] I am test_fn\n");
}

PyMODINIT_FUNC
PyInit__callback(void)
{
    PyObject *m;

    // We need to initialize PyArray_API (table of function pointers)
    // in every translation unit (separate .c file).
    // See the details in the accepted solution here:
    // https://stackoverflow.com/q/47026900/1095202
    import_array();

    if (PyType_Ready(&PythonWrapperForCCallbackType) < 0) {
        fprintf(stderr, "[_callback] Type is not ready\n");
        return NULL;
    }

    m = PyModule_Create(&callbackmodule);
    if (m == NULL) {
        fprintf(stderr, "[_callback] Could not create module\n");
        return NULL;
    }

    if (PyModule_AddObject(m, "PythonWrapperForCCallback",
                           (PyObject *)&PythonWrapperForCCallbackType) < 0) {
        goto fail;
    }

    PyObject *capsule = PyCapsule_New((void *)test_fn_, "123", NULL);
    if (PyModule_AddObject(m, "capsule", capsule) < 0) {
        fprintf(stderr, "[_callback] Could not add stub capsule\n");
        Py_DECREF(capsule);
        goto fail;
    }

    return m;
fail:
    Py_DECREF(m);
    return NULL;
}

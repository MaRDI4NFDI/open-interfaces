import ctypes
from typing import NewType

import numpy as np

UInt = NewType("UInt", int)


OIF_INT = 1
OIF_FLOAT32 = 2
OIF_FLOAT64 = 3
OIF_FLOAT32_P = 4
OIF_FLOAT64_P = 5
OIF_STR = 6


class OIFArgType(ctypes.c_int):
    pass


class OIFArgs(ctypes.Structure):
    _fields_ = [
        ("num_args", ctypes.c_size_t),
        ("arg_types", ctypes.POINTER(OIFArgType)),
        ("arg_values", ctypes.POINTER(ctypes.c_void_p)),
    ]


class OIFArray(ctypes.Structure):
    _fields_ = [
        ("nd", ctypes.c_int),
        ("dimensions", ctypes.POINTER(ctypes.c_int)),
        ("data", ctypes.POINTER(ctypes.c_char)),
    ]


_lib_dispatch = ctypes.PyDLL("./liboif_dispatch.so")


class OIFBackend:
    def __init__(self, handle):
        self.handle = handle

    def call(self, method, user_args, out_user_args):
        num_args = len(user_args)
        arg_types = []
        arg_values = []
        for arg in user_args:
            if isinstance(arg, int):
                arg_values.append(ctypes.c_void_p(ctypes.c_int(arg)))
                arg_types.append(OIF_INT)
            elif isinstance(arg, float):
                arg_p = ctypes.pointer(ctypes.c_double(arg))
                arg_void_p = ctypes.cast(arg_p, ctypes.c_void_p)
                arg_values.append(arg_void_p)
                arg_types.append(OIF_FLOAT64)
            elif isinstance(arg, np.ndarray):
                print("Warning: we assume that dtype is np.float64")
                arg_p = ctypes.pointer(arg.data.as_type(ctypes.c_void_p))
                arg_p_void = ctypes.cast(arg_p, ctypes.c_void_p)
                arg_values.append(arg_p_void)
                arg_types.append(OIF_FLOAT64_P)
            else:
                raise ValueError("Cannot handle argument type")

        arg_types_ctypes = ctypes.cast(
            (ctypes.c_int * len(arg_types))(*arg_types), ctypes.POINTER(OIFArgType)
        )
        arg_values_ctypes = ctypes.cast(
            (ctypes.c_void_p * len(arg_values))(*arg_values),
            ctypes.POINTER(ctypes.c_void_p),
        )
        args_packed = OIFArgs(num_args, arg_types_ctypes, arg_values_ctypes)

        num_out_args = len(out_user_args)
        out_arg_types = []
        out_arg_values = []
        for arg in out_user_args:
            if isinstance(arg, int):
                out_arg_values.append(ctypes.c_void_p(ctypes.c_int(arg)))
                out_arg_types.append(OIF_INT)
            elif isinstance(arg, float):
                arg_p = ctypes.pointer(ctypes.c_double(arg))
                arg_void_p = ctypes.cast(arg_p, ctypes.c_void_p)
                out_arg_values.append(arg_void_p)
                out_arg_types.append(OIF_FLOAT64)
            elif isinstance(arg, np.ndarray):
                print("Warning: we assume that dtype is np.float64")
                nd = arg.ndim
                dimensions = (ctypes.c_int * len(arg.shape))(*arg.shape)
                data = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_char))

                oif_array = OIFArray(nd, dimensions, data)
                oif_array_p = ctypes.cast(ctypes.byref(oif_array), ctypes.c_void_p)
                oif_array_p_p = ctypes.cast(ctypes.byref(oif_array_p), ctypes.c_void_p)
                out_arg_values.append(oif_array_p_p)
                out_arg_types.append(OIF_FLOAT64_P)
            else:
                raise ValueError("Cannot handle argument type")

        out_arg_types_ctypes = ctypes.cast(
            (ctypes.c_int * len(out_arg_types))(*out_arg_types),
            ctypes.POINTER(OIFArgType),
        )
        out_arg_values_ctypes = ctypes.cast(
            (ctypes.c_void_p * len(out_arg_values))(*out_arg_values),
            ctypes.POINTER(ctypes.c_void_p),
        )
        out_packed = OIFArgs(num_out_args, out_arg_types_ctypes, out_arg_values_ctypes)

        call_interface_method = wrap_c_function(
            _lib_dispatch,
            "call_interface_method",
            ctypes.c_int,
            [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.POINTER(OIFArgs),
                ctypes.POINTER(OIFArgs),
            ],
        )
        status = call_interface_method(
            self.handle,
            method.encode(),
            ctypes.byref(args_packed),
            ctypes.byref(out_packed),
        )

        if status != 0:
            raise RuntimeError("Could not execute interface method")

        # result = []
        # print("Output arguments after call_interface_method")
        # for typ, val in zip(out_arg_types, out_arg_values):
        #     if typ == OIF_FLOAT64_P:
        #         result.append(out_arg_values)

        return 0


def init_backend(backend: str, interface: str, major: UInt, minor: UInt):
    load_backend = wrap_c_function(
        _lib_dispatch,
        "load_backend_by_name",
        ctypes.c_uint,
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint],
    )
    handle = load_backend(backend.encode(), interface.encode(), major, minor)
    return OIFBackend(handle)


def wrap_c_function(lib, funcname, restype, argtypes):
    if isinstance(argtypes, list):
        if len(argtypes) == 1:
            assert argtypes[0] is not None, "For func(void) pass [] or None, not [None]"
    elif argtypes is not None:
        raise ValueError("Argument `argtypes` must be list or None")
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

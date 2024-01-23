import ctypes
from typing import Callable, NewType, Union

import numpy as np

UInt = NewType("UInt", int)


OIF_INT = 1
OIF_FLOAT32 = 2
OIF_FLOAT64 = 3
OIF_FLOAT32_P = 4
OIF_ARRAY_F64 = 5
OIF_STR = 6
OIF_CALLBACK = 7


_lib_dispatch = ctypes.PyDLL("liboif_dispatch.so")


class OIFArgType(ctypes.c_int):
    pass


class OIFArgs(ctypes.Structure):
    _fields_ = [
        ("num_args", ctypes.c_size_t),
        ("arg_types", ctypes.POINTER(OIFArgType)),
        ("arg_values", ctypes.POINTER(ctypes.c_void_p)),
    ]


class OIFArrayF64(ctypes.Structure):
    _fields_ = [
        ("nd", ctypes.c_int),
        ("dimensions", ctypes.POINTER(ctypes.c_long)),
        ("data", ctypes.POINTER(ctypes.c_double)),
    ]


class OIFCallback(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_int),
        ("fn_p_py", ctypes.c_void_p),
        ("fn_p_c", ctypes.c_void_p),
    ]


# class OIFCallback(ctypes.Structure):
#     _fields_ = [
#         ("fn", ctypes.CFUNCTYPE),
#         ("num_args", ctypes.c_size_t),
#         ("argtypes", ctypes.POINTER(OIFArgType)),
#         ("restype", OIFArgType),
#     ]


def make_oif_callback(
    fn: Callable, argtypes: list[OIFArgType], restype: OIFArgType
) -> OIFCallback:
    # id returns function pointer. Yes, I am also shocked.
    # Docstring for `id` says: "CPython uses the object's memory address".
    fn_p_py = id(fn)

    ctypes_argtypes: list = []
    for argt in argtypes:
        if argt == OIF_FLOAT64:
            ctypes_argtypes.append(ctypes.c_double)
        elif argt == OIF_ARRAY_F64:
            ctypes_argtypes.append(ctypes.POINTER(OIFArrayF64))
        else:
            raise ValueError(f"Cannot convert argument type {argt}")

    ctypes_restype: Union[type[ctypes.c_int], type[ctypes.c_double]]
    if restype == OIF_INT:
        ctypes_restype = ctypes.c_int
    elif restype == OIF_FLOAT64:
        ctypes_restype = ctypes.c_double
        # TODO: Add c_void_p type
        # Here there is a discussion why one can use only simple
        # predefined pointer types as restype:
        # https://stackoverflow.com/q/33005127
    else:
        raise ValueError(f"Cannot convert type '{restype}'")

    fn_t = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_argtypes)
    wrapper_fn = fn_t(make_oif_wrapper(fn, argtypes, restype))

    fn_p_c = ctypes.cast(wrapper_fn, ctypes.c_void_p)
    assert fn_p_c.value is not None
    # TODO: Replace magic constant with OIF_LANG_PYTHON (= 3)
    oifcallback = OIFCallback(3, fn_p_py, fn_p_c)
    return oifcallback


def make_oif_wrapper(fn: Callable, arg_types: list, restype):
    """Call `fn` converting OIF data types to native Python data types."""

    def wrapper(*arg_values):
        py_arg_values = []
        for i, (t, v) in enumerate(zip(arg_types, arg_values)):
            if t == OIF_INT:
                py_arg_values.append(v)
            elif t == OIF_FLOAT64:
                py_arg_values.append(v)
            elif t == OIF_ARRAY_F64:
                # v is a ctypes pointer to OIFArrayF64 struct.
                py_arg_values.append(
                    np.ctypeslib.as_array(
                        v.contents.data,
                        shape=[v.contents.dimensions[i] for i in range(v.contents.nd)],
                    )
                )
            else:
                raise ValueError("Unsupported data type")

        result = fn(*py_arg_values)
        if result is None:
            assert restype == OIF_INT
            result = 0
        return result

    return wrapper


class OIFPyBinding:
    def __init__(self, implh, interface, impl):
        self.implh = implh
        self.interface = interface
        self.impl = impl

    def call(self, method, user_args, out_user_args):
        num_args = len(user_args)
        arg_types = []
        arg_values = []
        for arg in user_args:
            if isinstance(arg, int):
                argp = ctypes.pointer(ctypes.c_int(arg))
                arg_void_p = ctypes.cast(argp, ctypes.c_void_p)
                arg_values.append(arg_void_p)
                arg_types.append(OIF_INT)
            elif isinstance(arg, float):
                arg_p = ctypes.pointer(ctypes.c_double(arg))
                arg_void_p = ctypes.cast(arg_p, ctypes.c_void_p)
                arg_values.append(arg_void_p)
                arg_types.append(OIF_FLOAT64)
            elif isinstance(arg, np.ndarray) and arg.dtype == np.float64:
                assert arg.dtype == np.float64
                nd = arg.ndim
                dimensions = (ctypes.c_long * len(arg.shape))(*arg.shape)
                data = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                oif_array = OIFArrayF64(nd, dimensions, data)
                oif_array_p = ctypes.cast(ctypes.pointer(oif_array), ctypes.c_void_p)
                oif_array_p_p = ctypes.cast(
                    ctypes.pointer(oif_array_p), ctypes.c_void_p
                )
                arg_values.append(oif_array_p_p)
                arg_types.append(OIF_ARRAY_F64)
            elif isinstance(arg, OIFCallback):
                argp = ctypes.pointer(arg)
                arg_values.append(ctypes.cast(argp, ctypes.c_void_p))
                arg_types.append(OIF_CALLBACK)
            else:
                raise ValueError(f"Cannot convert argument {arg} of type{type(arg)}")

        in_arg_types_ctypes = ctypes.cast(
            (ctypes.c_int * len(arg_types))(*arg_types), ctypes.POINTER(OIFArgType)
        )
        in_arg_values_ctypes = ctypes.cast(
            (ctypes.c_void_p * len(arg_values))(*arg_values),
            ctypes.POINTER(ctypes.c_void_p),
        )
        in_args_packed = OIFArgs(num_args, in_arg_types_ctypes, in_arg_values_ctypes)

        num_out_args = len(out_user_args)
        out_arg_types = []
        out_arg_values = []
        for arg in out_user_args:
            if isinstance(arg, int):
                argp = ctypes.pointer(ctypes.c_int(arg))
                arg_void_p = ctypes.cast(argp, ctypes.c_void_p)
                out_arg_values.append(arg_void_p)
                out_arg_types.append(OIF_INT)
            elif isinstance(arg, float):
                arg_p = ctypes.pointer(ctypes.c_double(arg))
                arg_void_p = ctypes.cast(arg_p, ctypes.c_void_p)
                out_arg_values.append(arg_void_p)
                out_arg_types.append(OIF_FLOAT64)
            elif isinstance(arg, np.ndarray) and arg.dtype == np.float64:
                nd = arg.ndim
                dimensions = (ctypes.c_long * len(arg.shape))(*arg.shape)
                data = arg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                oif_array = OIFArrayF64(nd, dimensions, data)
                oif_array_p = ctypes.cast(ctypes.pointer(oif_array), ctypes.c_void_p)
                oif_array_p_p = ctypes.cast(
                    ctypes.pointer(oif_array_p), ctypes.c_void_p
                )
                out_arg_values.append(oif_array_p_p)
                out_arg_types.append(OIF_ARRAY_F64)
            else:
                raise ValueError(f"Cannot convert argument {arg} of type{type(arg)}")

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
            self.implh,
            method.encode(),
            ctypes.byref(in_args_packed),
            ctypes.byref(out_packed),
        )

        if status != 0:
            raise RuntimeError("Could not execute interface method")

        return 0


def init_impl(interface: str, impl: str, major: UInt, minor: UInt):
    load_interface_impl = wrap_c_function(
        _lib_dispatch,
        "load_interface_impl",
        ctypes.c_int,
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint],
    )
    implh = load_interface_impl(interface.encode(), impl.encode(), major, minor)
    if implh < 0:
        raise RuntimeError("Cannot initialize backend")
    return OIFPyBinding(implh, interface, impl)


def unload_impl(binding: OIFPyBinding):
    unload_interface_impl = wrap_c_function(
        _lib_dispatch,
        "unload_interface_impl",
        ctypes.c_int,
        [ctypes.c_int],
    )
    status = unload_interface_impl(binding.implh)
    if status != 0:
        raise RuntimeError(
            f"Could not unload implementation '{binding.impl}' "
            f"of the interface '{binding.interface}' correctly"
        )


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

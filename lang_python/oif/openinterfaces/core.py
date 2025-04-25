import ctypes
import os
import site
from io import BytesIO
from typing import Callable, NewType, Union

import _conversion
import msgpack
import numpy as np

UInt = NewType("UInt", int)


OIF_INT = 1
OIF_FLOAT32 = 2
OIF_FLOAT64 = 3
OIF_FLOAT32_P = 4
OIF_ARRAY_F64 = 5
OIF_STR = 6
OIF_CALLBACK = 7
OIF_USER_DATA = 8
OIF_CONFIG_DICT = 9


OIF_LANG_C = 1
OIF_LANG_CXX = 2
OIF_LANG_PYTHON = 3
OIF_LANG_JULIA = 4
OIF_LANG_R = 5
OIF_LANG_COUNT = 6

_site_packages = site.getsitepackages()[-1]

# Add path to Python implementations to the environment variable.
path = os.path.join(_site_packages, "oif", "data")
if "OIF_IMPL_PATH" not in os.environ:
    os.environ["OIF_IMPL_PATH"] = path
else:
    os.environ["OIF_IMPL_PATH"] += os.pathsep + path

# We need to check if the library is in the site-packages directory
# because this is the only place I could install this library
# using `scikit-build-core` as a Python packaging build system.
if os.path.isfile(os.path.join(_site_packages, "lib", "liboif_dispatch.so")):
    _lib_dispatch = ctypes.PyDLL(
        os.path.join(_site_packages, "lib", "liboif_dispatch.so")
    )
else:
    _lib_dispatch = ctypes.PyDLL("liboif_dispatch.so")

elapsed = 0.0

elapsed_call = 0.0


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


class OIFUserData(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_int),
        ("c", ctypes.c_void_p),
        ("py", ctypes.c_void_p),
    ]


class OIFConfigDict(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # Type for sanity checks (OIF_CONFIG_DICT)
        ("src", ctypes.c_int),  # one of OIF_LANG_* constants
        ("size", ctypes.c_size_t),  # Current number of elements in the map
        ("buffer", ctypes.c_char_p),  # Buffer that is used by the pc
        ("buffer_length", ctypes.c_size_t),  # Buffer length, unsurprisingly
        ("py_object", ctypes.c_void_p),  # Python dictionary
    ]


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
        elif argt == OIF_USER_DATA:
            ctypes_argtypes.append(ctypes.c_void_p)
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
    c_wrapper_fn = fn_t(_make_c_func_wrapper_from_py_callable(fn, argtypes, restype))

    fn_p_c = ctypes.cast(c_wrapper_fn, ctypes.c_void_p)
    assert fn_p_c.value is not None
    oifcallback = OIFCallback(OIF_LANG_PYTHON, fn_p_py, fn_p_c)
    return oifcallback


def _make_c_func_wrapper_from_py_callable(fn: Callable, arg_types: list, restype):
    """Call `fn` converting OIF data types to native Python data types."""

    def _pyobject_from_pointer(v):
        if v is None:
            return v
        else:
            return ctypes.cast(v, ctypes.py_object).value

    type_conversion = {
        OIF_INT: lambda v: v,
        OIF_FLOAT64: lambda v: v,
        # v is a ctypes pointer to OIFArrayF64 struct.
        # v_p is a raw C pointer
        # Initially this block of code used numpy.ctypeslib.as_array
        # to obtain a NumPy array for existing C buffer, but somehow
        # it works very slowly, so we get NumPy arrays now using
        # C extension.
        OIF_ARRAY_F64: lambda v: _conversion.numpy_array_from_oif_array_f64(
            ctypes.addressof(v.contents)
        ),
        # ctypes receives a pointer to PyObject as Python int,
        # and this is how we convert it back to PyObject.
        OIF_USER_DATA: _pyobject_from_pointer,
    }
    assert not (arg_types - type_conversion.keys())

    # lib_convert = ctypes.CDLL("liboif_lang_python_convert.so")
    # convert_fn = _wrap_c_function(
    #     lib_convert,
    #     "python_types_from_oif_types",
    #     ctypes.c_int,
    #     [ctypes.py_object, ctypes.py_object, ctypes.py_object],
    # )

    def wrapper(*arg_values):
        py_arg_values = [None] * len(arg_types)
        for i, (t, v) in enumerate(zip(arg_types, arg_values)):
            py_arg_values[i] = type_conversion[t](v)
        # convert_fn(py_arg_values, arg_values, arg_types)

        result = fn(*py_arg_values)
        if result is None:
            result = 0
        return result

    return wrapper


def make_oif_user_data(data: object) -> OIFUserData:
    return OIFUserData(OIF_LANG_PYTHON, None, ctypes.c_void_p(id(data)))


def make_oif_config_dict(arg: dict) -> OIFConfigDict:
    buffer = BytesIO()
    # buffer.write(msgpack.packb(len(arg.keys()), use_bin_type=True))
    for k in arg.keys():
        v1 = k
        v2 = arg[k]
        if isinstance(v2, int) or isinstance(v2, float) or isinstance(v2, str):
            pass
        else:
            raise TypeError("Supported types for dictionaries are: int, float, str")

        buffer.write(msgpack.packb(v1, use_bin_type=True))
        buffer.write(msgpack.packb(v2, use_bin_type=True))
    buffer.seek(0)

    obj = OIFConfigDict(
        OIF_CONFIG_DICT,
        OIF_LANG_PYTHON,
        0,
        buffer.getvalue(),
        len(buffer.getvalue()),
        id(arg),
    )

    return obj


class OIFPyBinding:
    def __init__(self, implh, interface, impl):
        self.implh = implh
        self.interface = interface
        self.impl = impl

        self._call_interface_impl = _wrap_c_function(
            _lib_dispatch,
            "call_interface_impl",
            ctypes.c_int,
            [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.POINTER(OIFArgs),
                ctypes.POINTER(OIFArgs),
            ],
        )

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
            elif isinstance(arg, str):
                arg_p = ctypes.pointer(ctypes.c_char_p(arg.encode()))
                arg_void_p = ctypes.cast(arg_p, ctypes.c_void_p)
                arg_values.append(arg_void_p)
                arg_types.append(OIF_STR)
            elif isinstance(arg, OIFCallback):
                argp = ctypes.pointer(arg)
                arg_values.append(ctypes.cast(argp, ctypes.c_void_p))
                arg_types.append(OIF_CALLBACK)
            elif isinstance(arg, OIFUserData):
                argp = ctypes.pointer(arg)
                arg_values.append(ctypes.cast(argp, ctypes.c_void_p))
                arg_types.append(OIF_USER_DATA)
            elif isinstance(arg, dict):
                arg_config_dict = make_oif_config_dict(arg)
                arg_config_dict_p = ctypes.pointer(arg_config_dict)
                arg_config_dict_p_p = ctypes.cast(
                    ctypes.pointer(arg_config_dict_p), ctypes.c_void_p
                )
                arg_values.append(arg_config_dict_p_p)
                arg_types.append(OIF_CONFIG_DICT)
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

        status = self._call_interface_impl(
            self.implh,
            method.encode(),
            ctypes.byref(in_args_packed),
            ctypes.byref(out_packed),
        )

        if status != 0:
            raise RuntimeError(f"Error occurred while executing method '{method}'")

        return 0


def load_impl(interface: str, impl: str, major: UInt, minor: UInt):
    load_interface_impl = _wrap_c_function(
        _lib_dispatch,
        "load_interface_impl",
        ctypes.c_int,
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint],
    )
    implh = load_interface_impl(interface.encode(), impl.encode(), major, minor)
    if implh < 0:
        raise RuntimeError(
            "Error occurred "
            f"during initialization of the implementation '{impl}' "
            f"for the interface '{interface}'"
        )
    return OIFPyBinding(implh, interface, impl)


def unload_impl(binding: OIFPyBinding):
    unload_interface_impl = _wrap_c_function(
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


def _wrap_c_function(lib, funcname, restype, argtypes):
    if isinstance(argtypes, list):
        if len(argtypes) == 1:
            assert argtypes[0] is not None, "For func(void) pass [] or None, not [None]"
    elif argtypes is not None:
        raise ValueError("Argument `argtypes` must be list or None")
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

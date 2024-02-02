import ctypes
import time

import _callback
import numpy as np
from oif.core import OIF_ARRAY_F64, OIF_FLOAT64, OIF_INT, OIFArrayF64


class Callback:
    def __init__(self, fn_p, id: str):
        get_pointer_fn = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer_fn.restype = ctypes.c_void_p
        raw_pointer = get_pointer_fn(ctypes.py_object(fn_p), id.encode())

        self.fn_capsule = fn_p
        self.arg_types = [OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64]

        c_func_wrapper = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(OIFArrayF64),
            ctypes.POINTER(OIFArrayF64),
        )
        self.fn_p_py = c_func_wrapper(raw_pointer)

    def __call__(self, *args):
        print(f"args are: {args}")
        return _callback.call_c_fn_from_python(self.fn_capsule, args)
        c_args = []
        for i, (t, v) in enumerate(zip(self.arg_types, args)):
            if t == OIF_INT:
                c_args.append(ctypes.c_int(v))
            elif t == OIF_FLOAT64:
                c_args.append(ctypes.c_double(v))
            elif t == OIF_ARRAY_F64:
                assert v.dtype == np.float64
                nd = v.ndim
                dimensions = (ctypes.c_long * len(v.shape))(*v.shape)
                data = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                oif_array = OIFArrayF64(nd, dimensions, data)
                c_args.append(ctypes.pointer(oif_array))

        return self.fn_p_py(*c_args)

import ctypes

from enum import Enum


class OIFArgTypeEnum(Enum):
    OIF_INT = 1
    OIF_FLOAT32 = 2
    OIF_FLOAT64 = 3
    OIF_STR = 4


class OIFArgType(ctypes.c_int):
    _values_ = [(item.value, item.name) for item in OIFArgTypeEnum]


class OIFArgTypes(ctypes.Structure):
    _fields_ = [
        ("arg_types", ctypes.POINTER(OIFArgType)),
        ("args", ctypes.c_void_p),
        ("num_args", ctypes.c_size_t)
    ]


class OIFBackend:
    def __init__(self, handle):
        self.handle = handle
        #self.dispatch = ctypes.CDLL("oif_dispatch.so")

    def call(self, method, user_args, out):
        arg_types = []
        args = []
        num_types = len(user_args)
        for arg in user_args:
            if type(arg) == OIFArgType.OIF_INT:
                args.append(ctypes.c_int(arg))
                arg_types.append(OIFArgType.OIF_INT)

        args_packed = OIFArgTypes(arg_types, args, num_args)
        #self.dispatch.call_interface_method(
        #    self.handle,
        #    method,
        #    args_packed,
        #    out_packed
        #)

        return 42

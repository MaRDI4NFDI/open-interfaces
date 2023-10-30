import numpy as np
from oif.core import OIF_ARRAY_F64, OIF_INT, OIFPyBinding, init_impl, wrap_py_func


class IVP:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = init_impl("ivp", impl, 1, 0)
        self.s = None

    def set_rhs_fn(self, rhs_fn):
        wrapper = wrap_py_func(rhs_fn, (OIF_ARRAY_F64, OIF_ARRAY_F64), OIF_INT)
        self._binding.call("set_rhs_fn", (wrapper,), ())

    def set_initial_value(self, t0, y0):
        t0 = float(t0)
        y0 = np.asarray(y0)
        self._binding.call("set_initial_value", (t0, y0), ())

    def integrate(self, t, y):
        self._binding.call("integrate", (t,), (y,))

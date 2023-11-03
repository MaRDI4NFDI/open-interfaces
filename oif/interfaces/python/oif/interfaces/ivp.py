import numpy as np
from oif.core import OIF_ARRAY_F64, OIF_INT, OIFPyBinding, init_impl, wrap_py_func


class IVP:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = init_impl("ivp", impl, 1, 0)
        self.s = None
        self.y0: np.ndarray
        self.y: np.ndarray

    def set_rhs_fn(self, rhs_fn):
        x = np.array([1.0, 2.0])
        if len(rhs_fn(2.718, x)) != len(x):
            raise ValueError("Right-hand-side function has signature problems")

        # TODO: Think more about the signature of rhs function in C
        wrapper = wrap_py_func(rhs_fn, (OIF_ARRAY_F64, OIF_ARRAY_F64), OIF_INT)
        self._binding.call("set_rhs_fn", (wrapper,), ())

    def set_initial_value(self, y0, t0):
        y0 = np.asarray(y0, dtype=np.float64)
        self.y0 = y0
        self.y = np.empty_like(y0)
        t0 = float(t0)
        self._binding.call("set_initial_value", (y0, t0), ())

    def integrate(self, t):
        self._binding.call("integrate", (t,), (self.y,))

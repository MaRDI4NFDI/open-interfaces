import ctypes

import numpy as np
from oif.core import (
    OIF_ARRAY_F64,
    OIF_FLOAT64,
    OIF_INT,
    OIFArrayF64,
    OIFPyBinding,
    init_impl,
    wrap_py_func,
)


class IVP:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = init_impl("ivp", impl, 1, 0)
        self.s = None
        self.N: int = 0
        self.y0: np.ndarray
        self.y: np.ndarray

    def set_rhs_fn(self, rhs_fn):
        if self.N <= 0:
            raise RuntimeError("'set_initial_value' must be called before 'set_rhs_fn'")
        x = np.random.random(size=(self.N,))
        if len(rhs_fn(2.718, x)) != len(x):
            raise ValueError("Right-hand-side function has signature problems")

        def rhs_fn_w(
            t: float, y: ctypes.POINTER(OIFArrayF64), y_dot: ctypes.POINTER(OIFArrayF64)
        ):
            assert y is not None
            assert y_dot is not None
            assert y.contents.dimensions[0] == self.N, (
                "Assertion that y->dimensions == N has failed. "
                f"Actual value of y->dimensions is {y.contents.dimensions}"
            )
            assert y_dot.contents.dimensions[0] == self.N
            y_np = np.ctypeslib.as_array(y.contents.data, shape=(self.N,))
            y_dot_np = np.ctypeslib.as_array(y_dot.contents.data, shape=(self.N,))
            y_dot_np[:] = rhs_fn(t, y_np)
            return 0

        self.wrapper = wrap_py_func(
            rhs_fn, rhs_fn_w, (OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64), OIF_INT
        )
        self._binding.call("set_rhs_fn", (self.wrapper,), ())

    def set_initial_value(self, y0, t0):
        y0 = np.asarray(y0, dtype=np.float64)
        self.y0 = y0
        self.y = np.empty_like(y0)
        self.N = len(self.y0)
        t0 = float(t0)
        self._binding.call("set_initial_value", (y0, t0), ())

    def integrate(self, t):
        self._binding.call("integrate", (t,), (self.y,))

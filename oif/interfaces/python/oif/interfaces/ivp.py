from typing import Callable

import numpy as np
from oif.core import (
    OIF_ARRAY_F64,
    OIF_FLOAT64,
    OIF_INT,
    OIF_USER_DATA,
    OIFPyBinding,
    load_impl,
    make_oif_callback,
    make_oif_user_data,
    unload_impl,
)

rhs_fn_t = Callable[[float, np.ndarray, np.ndarray, object], int]


class IVP:
    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("ivp", impl, 1, 0)
        self.s = None
        self.N: int = 0
        self.y0: np.ndarray
        self.y: np.ndarray

    def set_initial_value(self, y0: np.ndarray, t0: float):
        y0 = np.asarray(y0, dtype=np.float64)
        self.y0 = y0
        self.y = np.empty_like(y0)
        self.N = len(self.y0)
        t0 = float(t0)
        self._binding.call("set_initial_value", (y0, t0), ())

    def set_rhs_fn(self, rhs_fn: rhs_fn_t):
        if self.N <= 0:
            raise RuntimeError("'set_initial_value' must be called before 'set_rhs_fn'")

        self.wrapper = make_oif_callback(
            rhs_fn, (OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64, OIF_USER_DATA), OIF_INT
        )
        self._binding.call("set_rhs_fn", (self.wrapper,), ())

    def set_user_data(self, user_data: object):
        self.user_data = make_oif_user_data(user_data)
        self._binding.call("set_user_data", (self.user_data,), ())

    def set_tolerances(self, rtol: float, atol: float):
        self._binding.call("set_tolerances", (rtol, atol), ())

    def integrate(self, t: float):
        self._binding.call("integrate", (t,), (self.y,))

    def set_integrator(self, integrator_name: str, integrator_params: dict = {}):
        self._binding.call("set_integrator", (integrator_name, integrator_params), ())

    def print_stats(self):
        self._binding.call("print_stats", (), ())

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)

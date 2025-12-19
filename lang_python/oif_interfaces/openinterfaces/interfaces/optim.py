r"""This module defines the interface for solving minimization problems:

.. math::
    minimize_x f(x)

where :math:`f : \mathbb R^n \to \mathbb R`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from openinterfaces.core import (
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

ObjectiveFn: TypeAlias = Callable[[np.ndarray, object], int]
"""Signature of the objective function :math:`f(, y)`.

!!!!!!!!!    The function accepts four arguments:
!!!!!!!!!        - `t`: current time,
!!!!!!!!!        - `y`: state vector at time :math:`t`,
!!!!!!!!!        - `ydot`: output array to which the result of function evalutation is stored,
!!!!!!!!!        - `user_data`: additional context (user-defined data) that
!!!!!!!!!          must be passed to the function (e.g., parameters of the system).

"""


@dataclass
class OptimResult:
    status: int
    x: np.ndarray


class Optim:
    """Interface for solving optimization (minimization) problems.

    This class serves as a gateway to the implementations of the
    solvers for optimization problems.

    Parameters
    ----------
    impl : str
        Name of the desired implementation.

    Examples
    --------

    Let's solve the following initial value problem:

    .. math::
        y'(t) = -y(t), \\quad y(0) = 1.

    First, import the necessary modules:
    >>> import numpy as np
    >>> from oif.interfaces.ivp import IVP

    Define the right-hand side function:

    >>> def rhs(t, y, ydot, user_data):
    ...     ydot[0] = -y[0]
    ...     return 0  # No errors, optional

    Now define the initial condition:

    >>> y0, t0 = np.array([1.0]), 0.0

    Create an instance of the IVP solver using the implementation "jl_diffeq",
    which is an adapter to the `OrdinaryDiffeq.jl` Julia package:

    >>> s = IVP("jl_diffeq")

    We set the initial value, the right-hand side function, and the tolerance:

    >>> s.set_initial_value(y0, t0)
    >>> s.set_rhs_fn(rhs)
    >>> s.set_tolerances(1e-6, 1e-12)

    Now we integrate to time `t = 1.0` in a loop, outputting the current value
    of `y` with time step `0.1`:

    >>> t = t0
    >>> times = np.linspace(t0, t0 + 1.0, num=11)
    >>> for t in times[1:]:
    ...     s.integrate(t)
    ...     print(f"{t:.1f} {s.y[0]:.6f}")
    0.1 0.904837
    0.2 0.818731
    0.3 0.740818
    0.4 0.670320
    0.5 0.606531
    0.6 0.548812
    0.7 0.496585
    0.8 0.449329
    0.9 0.406570
    1.0 0.367879

    """

    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("optim", impl, 1, 0)
        self.x0: np.ndarray
        """Current value of the state vector."""
        self._N: int = 0
        self.status = -1
        self.x: np.ndarray

    def set_initial_guess(self, x0: np.ndarray):
        """Set initial guess for the optimization problem"""
        self.x0 = x0
        self._N = len(self.x0)
        self.x = np.empty((self._N,))
        self._binding.call("set_initial_guess", (x0,), ())

    def set_objective_fn(self, objective_fn: ObjectiveFn):
        self.wrapper = make_oif_callback(
            objective_fn, (OIF_ARRAY_F64, OIF_USER_DATA), OIF_FLOAT64
        )
        self._binding.call("set_objective_fn", (self.wrapper,), ())

    # def set_tolerances(self, rtol: float, atol: float):
    #     """Specify relative and absolute tolerances, respectively."""
    #     self._binding.call("set_tolerances", (rtol, atol), ())

    def set_user_data(self, user_data: object):
        """Specify additional data that will be used for right-hand side function."""
        self.user_data = make_oif_user_data(user_data)
        self._binding.call("set_user_data", (self.user_data,), ())

    # def set_integrator(self, integrator_name: str, integrator_params: dict = {}):
    #     """Set integrator, if the name is recognizable."""
    #     self._binding.call("set_integrator", (integrator_name, integrator_params), ())

    def minimize(self):
        """Integrate to time `t` and write solution to `y`."""
        self._binding.call(
            "minimize",
            (),
            (self.x,),
        )

        self.result = OptimResult(status=self.status, x=self.x)

        return self.result

    # def print_stats(self):
    #     """Print integration statistics."""
    #     self._binding.call("print_stats", (), ())

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)

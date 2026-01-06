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
    OIF_TYPE_ARRAY_F64,
    OIF_TYPE_F64,
    OIF_TYPE_INT,
    OIF_TYPE_STRING,
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


class Optim:
    r"""Interface for solving optimization (minimization) problems.

    This class serves as a gateway to the implementations of the
    solvers for optimization problems.

    Parameters
    ----------
    impl : str
        Name of the desired implementation.

    Examples
    --------

    Let's solve the following convex optimization problem:

    .. math::
        minimize \sum_{i = 1}^N x_i^2

    where the solution is :math:`[0, ..., 0]` as the problem is convex.

    First, import the necessary modules:
    >>> import numpy as np
    >>> from oif.interfaces.optim import Optim

    Define the objective function:

    >>> def objective_fn(x):
    ...     return np.sum(x**2)


    Create an instance of the optim solver using the implementation "scipy_optimize",
    which is an adapter to the `scipy.optimize` Python package:

    >>> s = Optim("scipy_optimize")

    We set the initial value, the right-hand side function, and the tolerance:

    >>> s.set_initial_guess([2.718, 3.142])
    >>> s.set_objective_fn(objective_fn)

    Now we solve the minimization problem and print the return status and message:

    >>> status, message = s.minimize()

    We can print the resultant minimizer by retrieving it from the solver:

    >>> print(f"Minimizer is {s.x}")

    """

    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("optim", impl, 1, 0)
        self.x0: np.ndarray
        """Current value of the state vector."""
        self._N: int = 0
        self.x: np.ndarray

    def set_initial_guess(self, x0: np.ndarray):
        """Set initial guess for the optimization problem"""
        self.x0 = x0
        self._N = len(self.x0)
        self.x = np.empty((self._N,))
        self._binding.call("set_initial_guess", (x0,), ())

    def set_objective_fn(self, objective_fn: ObjectiveFn):
        self.wrapper = make_oif_callback(
            objective_fn, (OIF_TYPE_ARRAY_F64, OIF_USER_DATA), OIF_TYPE_F64
        )
        self._binding.call("set_objective_fn", (self.wrapper,), ())

    def set_user_data(self, user_data: object):
        """Specify additional data that will be used for right-hand side function."""
        self.user_data = make_oif_user_data(user_data)
        self._binding.call("set_user_data", (self.user_data,), ())

    def set_method(self, method_name: str, method_params: dict = {}):
        """Set integrator, if the name is recognizable."""
        self._binding.call("set_method", (method_name, method_params), ())

    def minimize(self):
        """Integrate to time `t` and write solution to `y`."""
        status, message = self._binding.call(
            "minimize",
            (),
            (self.x,),
            (OIF_TYPE_INT, OIF_TYPE_STRING),
        )

        return status, message

    # def print_stats(self):
    #     """Print integration statistics."""
    #     self._binding.call("print_stats", (), ())

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)

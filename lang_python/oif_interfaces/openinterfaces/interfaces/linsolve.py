"""This module defines the interface for solving linear systems of equations.

Problems to be solved are of the form:

    .. math::
        A x = b,

where :math:`A` is a square matrix and :math:`b` is a vector.
"""

import numpy as np
from openinterfaces.core import OIFPyBinding, load_impl, unload_impl


class Linsolve:
    """Interface for solving linear systems of equations.

    This class serves as a gateway to the implementations of the
    linear algebraic solvers.

    Parameters
    ----------
    impl : str
        Name of the desired implementation.

    """

    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("linsolve", impl, 1, 0)

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve the linear system of equations :math:`A x = b`.

        Parameters
        ----------
        A : np.ndarray of shape (n, n)
            Coefficient matrix.
        b : np.ndarray of shape (n,)
            Right-hand side vector.

        Returns
        -------
        np.ndarray
            Result of the linear system solution after the invocation
            of the `solve` method.

        """
        result = np.empty((A.shape[1]))
        self._binding.call("solve_lin", (A, b), (result,))
        return result

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)

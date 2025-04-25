"""This module defines the interface for solving a quadratic equation.

The quadratic equation is of the form:

.. math::
    a x^2 + b x + c = 0,

where :math:`a`, :math:`b`, and :math:`c` are the coefficients of the equation.

Of course, this is not very useful in scientific context to invoke
such a solver.

It was developed as a prototype to ensure that the envisioned architecture
of Open Interfaces is feasible.
It is used as a simple text case as well.

"""

import numpy as np
from openinterfaces.core import OIFPyBinding, load_impl, unload_impl


class QEQ:
    """Interface for solving quadratic equations.

    This class serves as a gateway to the implementations of the
    of the quadratic-equation solvers.

    Example
    -------

    Let's solve the following quadratic equation:

        .. math::
            x^2 + 2 x + 1 = 0.

    First, import the necessary modules:

    >>> from oif.interfaces.qeq import QEQ

    Define the coefficients of the equation:

    >>> a, b, c = 1.0, 2.0, 1.0

    Create an instance of the solver:

    >>> s = QEQ("py_qeq_solver")

    Solve the equation:

    >>> result = s.solve(a, b, c)
    >>> print(result)
    [-1. -1.]

    """

    def __init__(self, impl: str):
        self._binding: OIFPyBinding = load_impl("qeq", impl, 1, 0)

    def solve(self, a: float, b: float, c: float):
        """Solve the quadratic equation :math:`a x^2 + b x + c = 0`."""
        a = float(a)
        b = float(b)
        c = float(c)
        result = np.array([11.0, 22.0])
        self._binding.call("solve_qeq", (a, b, c), (result,))
        return result

    def __del__(self):
        if hasattr(self, "_binding"):
            unload_impl(self._binding)

import math

import numpy as np
from openinterfaces._impl.interfaces.qeq import QeqInterface


class QeqSolver(QeqInterface):
    def solve_qeq(self, a: float, b: float, c: float, roots: np.ndarray) -> int:
        """Solve quadratic equation ax**2 + bx + c = 0.

        Assumes that the output array `roots` always has two elements,
        so if the roots are repeated, they both will be still present.

        Parameters
        ----------
        a, b, c : float
            Coefficients.
        roots: np.ndarray[2]
            Array to which both roots are written.

        Returns
        -------
        int
            Status code.
        """
        if a == 0.0:
            roots[0] = -c / b
            roots[1] = -c / b
        else:
            D = b**2 - 4 * a * c
            if b > 0:
                roots[0] = (-b - math.sqrt(D)) / (2 * a)
                roots[1] = c / (a * roots[0])
            else:
                roots[0] = (-b + math.sqrt(D)) / (2 * a)
                roots[1] = c / (a * roots[0])

        return 0

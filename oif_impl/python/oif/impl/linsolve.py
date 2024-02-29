import abc

import numpy as np


class LinsolveInterface(abc.ABC):
    @abc.abstractmethod
    def solve_lin(self, A: np.ndarray, b: np.ndarray, result: np.ndarray) -> None:
        """Solve Ax = b, where `result` array is used for the solution."""

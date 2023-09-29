import abc

import numpy as np


class QeqInterface(abc.ABC):
    @abc.abstractmethod
    def solve_qeq(self, a: float, b: float, c: float, roots: np.ndarray) -> int:
        pass

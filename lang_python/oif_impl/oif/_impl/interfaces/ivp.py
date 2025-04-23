import abc
from typing import Callable, Dict, Union

import numpy as np


class IVPInterface(abc.ABC):
    @abc.abstractmethod
    def set_initial_value(self, y0: np.ndarray, t0: float) -> Union[int, None]:
        """Set initial value y(t0) = y0."""

    @abc.abstractmethod
    def set_rhs_fn(self, rhs: Callable) -> Union[int, None]:
        """Specify right-hand side function f."""

    @abc.abstractmethod
    def set_tolerances(self, rtol: float, atol: float) -> Union[int, None]:
        """Specify relative and absolute tolerances, respectively."""

    @abc.abstractmethod
    def set_user_data(self, user_data: object) -> Union[int, None]:
        """Specify additional data that will be used for right-hand side function."""

    @abc.abstractmethod
    def set_integrator(
        self, integrator_name: str, integrator_params: Dict
    ) -> Union[int, None]:
        """Set integrator, if the name is recognizable."""

    @abc.abstractmethod
    def integrate(self, t: float, y: np.ndarray) -> Union[int, None]:
        """Integrate to time `t` and write solution to `y`."""

    @abc.abstractmethod
    def print_stats(self):
        """Print integration statistics."""

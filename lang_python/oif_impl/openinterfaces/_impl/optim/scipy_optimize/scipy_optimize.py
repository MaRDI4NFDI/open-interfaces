from typing import Callable

import numpy as np
from openinterfaces._impl.interfaces.optim import OptimInterface
from scipy import optimize

_prefix = "scipy_optimize"


class ScipyOptimize(OptimInterface):
    """
    Solve optimization problem minimize_x f(x)

    """

    def __init__(self) -> None:
        self.N = 0  # Problem dimension.
        self.x0: np.ndarray | None = None
        self.objective_fn: Callable | None = None
        self.user_data: object | None = None
        self.with_user_data = False
        self.method_name: str | None = None
        self.method_params: str | None = None

    def set_initial_guess(self, x0: np.ndarray):
        _p = f"[{_prefix}::set_initial_guess]"

        self.x0 = x0
        self.N = len(x0)

    def set_user_data(self, user_data):
        self.user_data = user_data
        self.with_user_data = True

    def set_objective_fn(self, objective_fn):
        self.objective_fn = objective_fn

        x = np.random.random(size=(self.N,))
        msg = "Wrong signature for the objective function: "
        msg += "expected return value must be of `float64` type"

        if self.with_user_data:
            assert type(self.objective_fn(x, self.user_data)) in [
                float,
                np.float64,
            ], msg
        else:
            assert type(self.objective_fn(x)) in [float, np.float64], msg

    def set_method(self, method_name, method_params):
        available_options = optimize.show_options("minimize", method=method_name)

        for k in method_params.keys():
            if k not in available_options:
                raise ValueError(
                    f"Option '{k}' is not known for method '{method_name}'"
                )

        self.method_name = method_name
        self.method_params = method_params

    def minimize(self, out_x):
        if self.with_user_data:
            result = optimize.minimize(
                self.objective_fn,
                self.x0,
                args=self.user_data,
                method=self.method_name,
                options=self.method_params,
            )
        else:
            result = optimize.minimize(
                self.objective_fn,
                self.x0,
                method=self.method_name,
                options=self.method_params,
            )

        out_x[:] = result.x
        return (result.status, result.message)

    # def print_stats(self):
    #     print("WARNING: `scipy_ode` does not provide statistics")

    # def _rhs_fn_wrapper(self, t, y):
    #     """Callback that satisfies signature expected by Open Interfaces."""
    #     self.rhs(t, y, self.ydot, self.user_data)
    #     return self.ydot

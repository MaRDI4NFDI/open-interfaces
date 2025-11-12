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
        self.objective_fn = None  # Right-hand side function.
        self.N = 0  # Problem dimension.
        self.x0: np.ndarray
        self.objective_fn: Callable
        # self.s = None
        # self.user_data = None
        # self.integrator = "dopri5"
        # self.integrator_params = {"rtol": 1e-6, "atol": 1e-12}

    def set_initial_guess(self, x0: np.ndarray):
        _p = f"[{_prefix}::set_initial_guess]"

        self.x0 = x0
        self.N = len(x0)

    def set_objective_fn(self, objective_fn):
        self.objective_fn = objective_fn

        x = np.random.random(size=(self.N,))
        msg = "Wrong signature for the objective function: "
        msg += "expected return value must be of `float64` type"
        assert type(self.objective_fn(x)) in [float, np.float64], msg

        return 0

    # def set_tolerances(self, rtol, atol):
    #     if self.s is None:
    #         raise RuntimeError("`set_rhs_fn` must be called before `set_tolerances`")

    #     self.integrator_params = self.integrator_params | {
    #         "rtol": rtol,
    #         "atol": atol,
    #     }
    #     self.s.set_integrator(self.integrator, **self.integrator_params)
    #     if hasattr(self, "y0"):
    #         self.s.set_initial_value(self.y0, self.t0)
    #     return 0

    # def set_user_data(self, user_data):
    #     self.user_data = user_data

    def minimize(self):
        self.result = optimize.minimize(self.objective_fn, self.x0)
        return 0

    def retrieve_optim_result(self, x):
        x[:] = self.result.x

    def retrieve_minval(self)

    # def get_n_rhs_evals(self, n_rhs_evals_arr):
    #     n_rhs_evals_arr[0] = self.s._integrator.iwork[16]
    #     return 0

    # def set_integrator(self, integrator_name, integrator_params):
    #     if self.s is None:
    #         raise RuntimeError("`set_integrator` must be called after `set_rhs_fn`")
    #     integrator = integrate._ode.find_integrator(integrator_name)
    #     if integrator is None:
    #         raise RuntimeError(
    #             "`set_integrator` received unknown integrator "
    #             "name {:s}".format(integrator_name)
    #         )
    #     if integrator_params is not None:
    #         self.integrator_params = self.integrator_params | integrator_params
    #     try:
    #         self.s.set_integrator(integrator_name, **self.integrator_params)
    #     except TypeError:
    #         raise RuntimeError(
    #             "Provided options {} are not accepted by integrator {}".format(
    #                 self.integrator_params, integrator_name
    #             )
    #         )
    #     if hasattr(self, "y0"):
    #         self.s.set_initial_value(self.y0, self.t0)
    #     return 0

    # def print_stats(self):
    #     print("WARNING: `scipy_ode` does not provide statistics")

    # def _rhs_fn_wrapper(self, t, y):
    #     """Callback that satisfies signature expected by Open Interfaces."""
    #     self.rhs(t, y, self.ydot, self.user_data)
    #     return self.ydot

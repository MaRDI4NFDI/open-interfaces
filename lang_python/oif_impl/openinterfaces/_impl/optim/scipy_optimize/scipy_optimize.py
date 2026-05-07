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
        self.grad_fn: Callable | None = None
        self.user_data: object | None = None
        self.with_user_data = False
        self.method_name: str | None = None
        self.method_params: str | None = None
        self.grad_values: np.ndarray | None = None

    def set_initial_guess(self, x0: np.ndarray):
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

        assert type(self.objective_fn(x, self.user_data)) in [
            float,
            np.float64,
        ], msg

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn
        self.grad_values = np.empty((self.N,))

    def set_method(self, method_name, method_params):
        available_options = optimize.show_options(
            "minimize", method=method_name, disp=False
        )

        for k in method_params.keys():
            candidates = [f"\n{k} :", f"\n{k},"]
            found = False
            for c in candidates:
                if c in available_options:
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Method option '{k}' is not known for the method '{method_name}'"
                )

        self.method_name = method_name
        self.method_params = method_params

    def minimize(self, out_x):
        if self.x0 is None:
            raise RuntimeError("Method `set_initial_guess` must be called first")

        if self.objective_fn is None:
            raise RuntimeError("Method `set_objective_fn` must be called first")

        if len(self.x0) != len(out_x):
            raise ValueError(
                "Shapes of the output array and the initial-guess array differ"
            )

        grad_fn = None if self.grad_fn is None else self.grad_wrapper

        result = optimize.minimize(
            self.objective_fn,
            self.x0,
            args=(self.user_data,),
            jac=grad_fn,  # I wonder, why the gradient in scipy.optimize is `jac`?
            method=self.method_name,
            options=self.method_params,
        )

        out_x[:] = result.x
        print(result)
        return (result.status, result.message)

    def grad_wrapper(self, x, __):
        grad_values = np.empty_like(x)
        # grad_values = self.grad_values
        self.grad_fn(x, grad_values, self.user_data)
        return grad_values

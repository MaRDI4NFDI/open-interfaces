import numpy as np
from openinterfaces._impl.interfaces.ivp import IVPInterface
from scipy import integrate

_prefix = "scipy_ode"


class ScipyODE(IVPInterface):
    def __init__(self):
        self.rhs = None  # Right-hand side function.
        self.N = 0  # Problem dimension.
        self.s = None
        self.user_data = None
        self.integrator = "dopri5"
        self.integrator_params = {"rtol": 1e-6, "atol": 1e-12}

    def set_initial_value(self, y0: np.ndarray, t0: float):
        _p = f"[{_prefix}::set_initial_value]"

        if not isinstance(t0, float):
            raise ValueError(f"{_p} Argument `t0` must be floating-point number")

        self.y0 = y0
        self.t0 = t0
        self.N = len(y0)
        self.ydot = np.empty_like(y0)

    def set_rhs_fn(self, rhs):
        if self.N <= 0:
            raise RuntimeError("`set_initial_value` must be called before `set_rhs_fn`")

        self.rhs = rhs
        x = np.random.random(size=(self.N,))
        msg = "Wrong signature for the right-hand side function"
        assert len(self._rhs_fn_wrapper(42.0, x)) == len(x), msg

        self.s = integrate.ode(self._rhs_fn_wrapper).set_integrator(
            self.integrator, **self.integrator_params
        )
        self.s.set_initial_value(self.y0, self.t0)

        return 0

    def set_tolerances(self, rtol, atol):
        if self.s is None:
            raise RuntimeError("`set_rhs_fn` must be called before `set_tolerances`")

        self.integrator_params = self.integrator_params | {
            "rtol": rtol,
            "atol": atol,
        }
        self.s.set_integrator(self.integrator, **self.integrator_params)
        if hasattr(self, "y0"):
            self.s.set_initial_value(self.y0, self.t0)
        return 0

    def set_user_data(self, user_data):
        self.user_data = user_data

    def integrate(self, t, y):
        y[:] = self.s.integrate(t)
        assert self.s.successful()
        return 0

    def get_n_rhs_evals(self, n_rhs_evals_arr):
        n_rhs_evals_arr[0] = self.s._integrator.iwork[16]
        return 0

    def set_integrator(self, integrator_name, integrator_params):
        if self.s is None:
            raise RuntimeError("`set_integrator` must be called after `set_rhs_fn`")
        integrator = integrate._ode.find_integrator(integrator_name)
        if integrator is None:
            raise RuntimeError(
                "`set_integrator` received unknown integrator "
                "name {:s}".format(integrator_name)
            )
        if integrator_params is not None:
            self.integrator_params = self.integrator_params | integrator_params
        try:
            self.s.set_integrator(integrator_name, **integrator_params)
        except TypeError:
            raise RuntimeError(
                "Provided options {} are not accepted by integrator {}".format(
                    integrator_params, integrator_name
                )
            )
        if hasattr(self, "y0"):
            self.s.set_initial_value(self.y0, self.t0)
        return 0

    def print_stats(self):
        print("WARNING: `scipy_ode` does not provide statistics")

    def _rhs_fn_wrapper(self, t, y):
        """Callback that satisfies signature expected by Open Interfaces."""
        self.rhs(t, y, self.ydot, self.user_data)
        return self.ydot

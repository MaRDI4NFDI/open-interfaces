import numpy as np
from scipy import integrate

_prefix = "scipy_ode_dopri5"


class Dopri5:
    def __init__(self):
        self.rhs = None  # Right-hand side function.
        self.N = 0  # Problem dimension.

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
            "dopri5", atol=1e-15, rtol=1e-15, nsteps=1000
        )
        self.s.set_initial_value(self.y0, self.t0)

        return 0

    def integrate(self, t, y):
        y[:] = self.s.integrate(t)
        assert self.s.successful()
        return 0

    def _rhs_fn_wrapper(self, t, y):
        """Callback that satisfies scipy.ode.dopri5 expectations."""
        self.rhs(t, y, self.ydot)
        return self.ydot

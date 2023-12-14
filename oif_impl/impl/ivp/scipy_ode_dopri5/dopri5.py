import numpy as np
from scipy import integrate

_prefix = "scipy_ode_dopri5"


class Dopri5:
    def __init__(self):
        self.rhs = None

    def set_rhs_fn(self, rhs):
        self.rhs = rhs

        # x = np.array([3.0])
        # assert len(self.rhs(42.0, x)) == len(x)
        # print("after check")

        return 0

    def set_initial_value(self, y0: np.ndarray, t0: float):
        _p = f"[{_prefix}::set_initial_value]"

        if not isinstance(t0, float):
            raise ValueError(f"{_p} Argument `t0` must be floating-point number")

        if self.rhs is None:
            raise TypeError(f"{_p} Method `set_rhs_fn` must be called earlier")

        self.s = integrate.ode(self.rhs).set_integrator(
            "dopri5", atol=1e-15, rtol=1e-15, nsteps=1000
        )
        self.s.set_initial_value(y0, t0)

    def integrate(self, t, y):
        y[:] = self.s.integrate(t)
        assert self.s.successful()
        return 0

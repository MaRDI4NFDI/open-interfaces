import numpy as np
from scipy import integrate


class Dopri5:
    def __init__(self):
        self.rhs = None

    def set_rhs_fn(self, rhs):
        self.rhs = rhs

        return 0

    def set_initial_value(self, y0, t0):
        if not isinstance(t0, float):
            raise ValueError("Argument `t0` must be floating-point number")

        self.s = integrate.ode(self.rhs).set_integrator(
            "dopri5", atol=1e-15, rtol=1e-15
        )
        self.s.set_initial_value(y0, t0)

    def integrate(self, t, y):
        y[:] = self.s.integrate(t)
        assert self.s.successful()
        return 0

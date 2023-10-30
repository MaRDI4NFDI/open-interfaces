import numpy as np
from scipy import integrate


class Dopri5:
    def __init__(self):
        self.rhs = None

    def set_rhs_fn(self, rhs):
        self.rhs = rhs

        x = np.array([1.0, 2.0])
        assert len(self.rhs(x)) == len(x)

        return 0

    def set_initial_value(self, t0, y0):
        self.s = integrate.ode(self.rhs).set_integrator("dopri5")
        self.s.set_initial_value(y0, t0)

    def integrate(self, t, y):
        y[:] = self.s.integrate(t)
        return 0

from abc import ABC, abstractmethod

import numpy as np
import numpy.testing as npt
import pytest
from oif.interfaces.ivp import IVP


class IVPProblem(ABC):
    t0: float
    y0: np.ndarray

    @abstractmethod
    def rhs(self, t, y):
        """Right-hand side function for a system of ODEs: y'(t) = f(t, y)."""
        pass

    @abstractmethod
    def exact(self, t):
        """Return exact solution at time `t`."""
        pass


class ScalarExpDecayProblem(IVPProblem):
    t0 = 0.0
    y0 = np.array([1.0])

    def rhs(self, _, y):
        return -y

    def exact(self, t):
        return self.y0 * np.exp(-t)


class LinearOscillatorProblem(IVPProblem):
    t0 = 0.0
    y0 = np.array([1.0, 0.5])
    omega = np.pi

    def rhs(self, t, y):
        return np.array(
            [
                y[1],
                -(self.omega**2) * y[0],
            ]
        )

    def exact(self, t):
        y_exact = (
            self.y0[0] * np.cos(self.omega * t)
            + self.y0[1] * np.sin(self.omega * t) / self.omega
        )
        return y_exact


class TestIVPViaScipyODEDopri5Implementation:
    def test_1(self, s, p):
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-4)

    def test_3_test_accept_int_list_for_y0_and_int_for_t0(self, s, p):
        s.set_initial_value(list(p.y0), int(p.t0))
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-4)


@pytest.fixture(
    params=[
        "scipy_ode_dopri5",
        "sundials_cvode",
    ]
)
def s(request):
    """Instantiate IVP with the specified implementation."""
    return IVP(request.param)


@pytest.fixture(params=[ScalarExpDecayProblem(), LinearOscillatorProblem()])
def p(request):
    """Return instantiated IVPProblem subclasses."""
    return request.param

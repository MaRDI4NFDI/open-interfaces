from abc import ABC, abstractmethod

import numpy as np
import numpy.testing as npt
import pytest
from oif.interfaces.ivp import IVP
from scipy import optimize


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
        return np.array(
            [
                self.y0[0] * np.cos(self.omega * t)
                + self.y0[1] * np.sin(self.omega * t) / self.omega,
                -self.y0[0] * self.omega * np.sin(self.omega * t)
                + self.y0[1] * np.cos(self.omega * t),
            ]
        )


class OrbitEquationsProblem(IVPProblem):
    eps = 0.9
    t0 = 0.0
    y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])

    def rhs(self, t, y):
        r = np.sqrt(y[0] ** 2 + y[1] ** 2)
        return np.array(
            [
                y[2],
                y[3],
                -y[0] / r**3,
                -y[1] / r**3,
            ]
        )

    def exact(self, t):
        def f(u):
            return u - self.eps * np.sin(u) - t

        def df(u):
            return 1 - self.eps * np.cos(u)

        u, __, ier, msg = optimize.fsolve(
            f, 1.0, fprime=df, xtol=1e-15, full_output=True
        )
        assert ier == 1, msg
        u = u[0]  # Extract the scalar value from the array.
        eps = self.eps
        return np.array(
            [
                np.cos(u) - eps,
                np.sqrt(1 - eps**2) * np.sin(u),
                -np.sin(u) / (1 - eps * np.cos(u)),
                (np.sqrt(1 - eps**2) * np.cos(u)) / (1 - eps * np.cos(u)),
            ]
        )


class TestIVPViaScipyODEDopri5Implementation:
    def test_1(self, s, p):
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y)

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-10)

    def test_2_test_accept_int_list_for_y0_and_int_for_t0(self, s, p):
        s.set_initial_value(list(p.y0), int(p.t0))
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y)

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-10)


@pytest.fixture(
    params=[
        "scipy_ode_dopri5",
        "sundials_cvode",
    ]
)
def s(request):
    """Instantiate IVP with the specified implementation."""
    return IVP(request.param)


@pytest.fixture(
    params=[
        ScalarExpDecayProblem(),
        LinearOscillatorProblem(),
        OrbitEquationsProblem(),
    ]
)
def p(request):
    """Return instantiated IVPProblem subclasses."""
    return request.param

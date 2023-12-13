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


class TestIVPViaScipyODEDopri5Implementation:
    @pytest.fixture(
        params=[
            "scipy_ode_dopri5",
            "sundials_cvode",
        ]
    )
    def s(self, request):
        return IVP(request.param)

    def test_1(self, s):
        p = ScalarExpDecayProblem()
        s.set_rhs_fn(p.rhs)
        s.set_initial_value(p.y0, p.t0)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-4)

    def test_3_test_accept_int_list_for_y0_and_int_for_t0(self, s):
        p = ScalarExpDecayProblem()
        s.set_rhs_fn(p.rhs)
        s.set_initial_value(list(p.y0), p.t0)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=1e-4)

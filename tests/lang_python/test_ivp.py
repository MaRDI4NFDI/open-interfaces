import numpy as np
import numpy.testing as npt
import pytest
from oif.interfaces.ivp import IVP


def rhs(_, y):
    return -y


class TestIVPViaScipyODEDopri5Implementation:
    @pytest.fixture(
        params=[
            "scipy_ode_dopri5",
            "sundials_cvode",
        ]
    )
    def s(self, request):
        return IVP(request.param)

    def rhs_method(self, _, y):
        return -y

    def test_1(self, s):
        s.set_rhs_fn(rhs)
        t0 = 0.0
        y0 = np.array([1.0])
        s.set_initial_value(y0, t0)

        times = np.linspace(t0, t0 + 1, num=11)

        soln = [y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        # Solution is y(t) = e^(-t), so y(1) = 0.367879.
        npt.assert_allclose(soln[-1], 0.367879, rtol=1e-4)

    def test_2_accept_rhs_fn_method(self, s):
        s.set_rhs_fn(self.rhs_method)
        t0 = 0.0
        y0 = np.array([1.0])
        s.set_initial_value(y0, t0)

        times = np.linspace(t0, t0 + 1, num=11)

        soln = [y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        # Solution is y(t) = e^(-t), so y(1) = 0.367879.
        npt.assert_allclose(soln[-1], 0.367879, rtol=1e-4)

    def test_3_test_accept_int_list_for_y0_and_int_for_t0(self, s):
        s.set_rhs_fn(rhs)
        t0 = 0
        y0 = [1]
        s.set_initial_value(y0, t0)

        times = np.linspace(t0, t0 + 1, num=11)

        soln = [y0[0]]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y[0])

        # Solution is y(t) = e^(-t), so y(1) = 0.367879.
        npt.assert_allclose(soln[-1], 0.367879, rtol=1e-4)

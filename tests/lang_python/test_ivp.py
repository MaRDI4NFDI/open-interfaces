import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.testing as npt
import pytest
from openinterfaces.interfaces.ivp import IVP
from scipy import optimize


class IVPProblem(ABC):
    t0: float
    y0: np.ndarray

    @abstractmethod
    def rhs(self, t, y, ydot, user_data):
        r"""Right-hand side function for a system of ODEs: \dot y(t) = f(t, y)."""
        pass

    @abstractmethod
    def exact(self, t):
        """Return exact solution at time `t`."""
        pass


class ScalarExpDecayProblem(IVPProblem):
    """Problem :math:`y(t) = -y, y(0) = 1` with solution :math:`y(t)=exp(-t)`"""

    t0 = 0.0
    y0 = np.array([1.0])

    def rhs(self, _, y, ydot, __):
        ydot[:] = -y

    def exact(self, t):
        return self.y0 * np.exp(-t)


class LinearOscillatorProblem(IVPProblem):
    r"""Linear second-order equation of non-decaying oscillator.

    Second-order equation :math:`y''(t) + \omega^2 y(t) = 0` with given
    :math:`y(0) = y_0` and :math:`y'(0) = y'_0`, and parameter :math:`\omega`.

    Solution is given by::
        y(t) = y_0 * \cos(\omega t) + y'_0 \frac{\sin(\omega t) / \omega}
    """

    t0 = 0.0
    y0 = np.array([1.0, 0.5])
    omega = np.pi

    def rhs(self, _, y, ydot, __):
        ydot[0] = y[1]
        ydot[1] = -(self.omega**2) * y[0]

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
    """System of differential equations that describes movement of two bodies.

    This problem is problem 5 from Problem Class D from the paper:
    Hull, T. E. et al. 1972. Comparing numerical methods for ordinary
    differential equations. SIAM J. Numer. Anal., p. 620. doi:10.1137/0709052

    """

    eps = 0.9
    t0 = 0.0
    y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])

    def rhs(self, _, y, ydot, __):
        r = np.sqrt(y[0] ** 2 + y[1] ** 2)
        ydot[0] = y[2]
        ydot[1] = y[3]
        ydot[2] = -y[0] / r**3
        ydot[3] = -y[1] / r**3

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


class IVPProblemWithUserData(IVPProblem):
    t0 = 0.0
    y0 = np.array([0.0, 1.0])
    user_data_cache = None

    def rhs(self, _, y, ydot, user_data: tuple):
        self.user_data_cache = user_data
        a, b = user_data
        ydot[0] = y[0] + a
        ydot[1] = b * y[1]

    def exact(self, t):
        a, b = self.user_data_cache
        return np.array(
            [
                a * (np.exp(t) - 1),
                np.exp(b * t),
            ]
        )


class MildlyStiffODESystem(IVPProblem):
    t0 = 0.0
    y0 = np.array([1.0, 0.0])

    def rhs(self, t, y, ydot, user_data: tuple):
        ydot[0] = -16 * y[0] + 12 * y[1] + 16 * np.cos(t) - 13 * np.sin(t)
        ydot[1] = 12 * y[0] - 9 * y[1] - 11 * np.cos(t) + 9 * np.sin(t)

    def exact(self, t):
        return np.array([np.cos(t), np.sin(t)])


class TestIVP:
    def test_1(self, s, p):
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y)

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=2e-4)

    def test_2_test_accept_int_list_for_y0_and_int_for_t0(self, s, p):
        s.set_initial_value(list(p.y0), int(p.t0))
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        soln = [p.y0]
        for t in times[1:]:
            s.integrate(t)
            soln.append(s.y)

        npt.assert_allclose(soln[-1], p.exact(t1), rtol=2e-4)

    def test_3__more_stringent_tolerances_lead_to_smaller_errors(self, s, p):
        s.set_initial_value(list(p.y0), int(p.t0))
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        errors = []
        for tol in [1e-3, 1e-4, 1e-6, 1e-8]:
            s.set_initial_value(p.y0, p.t0)
            s.set_tolerances(tol, tol)
            for t in times[1:]:
                s.integrate(t)

            final_value = s.y
            true_value = p.exact(t1)
            error = np.linalg.norm(final_value - true_value)
            errors.append(error)

        for k in range(1, len(errors)):
            assert errors[k - 1] >= errors[k]

    def test_4__check_that_user_data_can_be_used(self, s):
        p = IVPProblemWithUserData()
        s.set_initial_value(list(p.y0), int(p.t0))
        s.set_user_data((12, 2.7))
        s.set_rhs_fn(p.rhs)
        s.set_tolerances(1e-6, 1e-8)

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=11)

        for t in times[1:]:
            s.integrate(t)

        final_value = s.y
        true_value = p.exact(t1)
        npt.assert_allclose(final_value, true_value, 1e-5, 1e-6)

    def test_5__check_that_we_can_set_integrator(self, solver_integrator, p):
        dt = 0.125

        s = IVP(solver_integrator[0])
        integrator_names = solver_integrator[1]
        p = OrbitEquationsProblem()

        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)
        s.set_tolerances(1e-2, 1e-2)

        s.integrate(p.t0 + dt)
        value_1 = s.y

        for integrator_name in integrator_names:
            print("Solver: ", solver_integrator[0])
            print("Integrator: ", integrator_name)
            s.set_integrator(integrator_name)
            s.set_initial_value(p.y0, p.t0)
            s.integrate(p.t0 + dt)
            value_2 = s.y

            npt.assert_allclose(value_1, value_2, 1e-1, 1e-1)

    def test_6__check_that_set_integrator_works_only_after_setting_rhs(self):
        s = IVP("scipy_ode")
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)

        with pytest.raises(RuntimeError):
            s.set_integrator("vode")

    def test_7__unknown_integrator_name__should_raise_exception(self):
        s = IVP("scipy_ode")
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        with pytest.raises(RuntimeError):
            s.set_integrator("i-am-not-known-integrator")


class TestIVPConfigDict:
    def test_1__config_dict_scipy_ode__should_accept_alright(self):
        s = IVP("scipy_ode")
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        params = {"method": "bdf", "order": 1}
        s.set_integrator("vode", params)

    def test_2__malformed_config_dict_scipy_ode__should_raise(self):
        s = IVP("scipy_ode")
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        with pytest.raises(RuntimeError):
            params = {"method": "bdf", "wrong-param-name": 1}
            s.set_integrator("vode", params)

    def test_3__malformed_config_dict_scipy_ode__should_raise(self):
        s = IVP("scipy_ode")
        p = ScalarExpDecayProblem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        with pytest.raises(RuntimeError):
            params = {"method": "bdf", "wrong-param-name": 1}
            s.set_integrator("vode", params)

    def test_4__config_dict_scipy_ode__should_be_stable_with_enough_nsteps(self):
        s = IVP("scipy_ode")
        p = MildlyStiffODESystem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        t1 = p.t0 + 1

        # We need to filter integration errors to avoid spurious output.
        warnings.simplefilter("error")
        # Set very small number of steps, so that integrator fails.
        with pytest.raises(RuntimeError):
            s.set_integrator("dopri5", {"nsteps": 1})
            s.integrate(t1)
        warnings.resetwarnings()

        # Set large number of steps, so that integrator succeeds.
        s.set_integrator("dopri5", {"nsteps": 10_000})
        s.integrate(t1)

    def test_5__config_dict_cvode__fails_when_max_num_steps_too_small(self):
        s = IVP("sundials_cvode")
        p = MildlyStiffODESystem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        s.set_integrator("bdf", {"max_num_steps": 50})

        t1 = p.t0 + 1
        times = np.linspace(p.t0, t1, num=2)

        with pytest.raises(RuntimeError):
            for t in times[1:]:
                s.integrate(t)

    def test_6__config_dict_cvode__fails_with_false_options(self):
        s = IVP("sundials_cvode")
        p = MildlyStiffODESystem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        # Should error on the unknown option.
        with pytest.raises(RuntimeError):
            s.set_integrator("bdf", {"max_num_steps_typo": 50})

    def test_7__config_dict_jl_diffeq__works(self):
        s = IVP("jl_diffeq")
        p = MildlyStiffODESystem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        s.set_integrator("Rosenbrock23", {"chunk_size": 10, "autodiff": False})

    def test_8__config_dict_jl_diffeq__fails_when_unknown_options(self):
        s = IVP("jl_diffeq")
        p = MildlyStiffODESystem()
        s.set_initial_value(p.y0, p.t0)
        s.set_rhs_fn(p.rhs)

        with pytest.raises(RuntimeError):
            s.set_integrator("DP5", {"unknown_option": 10_000})


@pytest.fixture(params=["scipy_ode", "sundials_cvode", "jl_diffeq"])
def s(request):
    """Instantiate IVP with the specified implementation."""
    print(f"IVP: {request.param}")
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
    print(f"Problem: {request.param}")
    return request.param


@pytest.fixture(
    params=[
        ("scipy_ode", ["vode", "lsoda", "dopri5", "dop853"]),
        ("sundials_cvode", ["bdf", "adams"]),
    ]
)
def solver_integrator(request):
    return request.param

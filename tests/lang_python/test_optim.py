import numpy as np
import pytest
from openinterfaces.interfaces.optim import Optim


# -----------------------------------------------------------------------------
# Problems
def convex_objective_fn(x):
    return np.sum(x**2)


def rosenbrock_objective_fn(x):
    """The Rosenbrock function with additional arguments"""
    a, b = (0.5, 1.0)
    return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b


@pytest.fixture(params=["scipy_optimize"])
def s(request):
    return Optim(request.param)


# -----------------------------------------------------------------------------
# Tests
def test__simple_convex_problem__converges(s):
    x0 = np.array([0.5, 0.6, 0.7])

    s.set_initial_guess(x0)
    s.set_objective_fn(convex_objective_fn)

    status, message = s.minimize()
    x = s.x

    assert status == 0
    assert len(x) == len(x0)
    assert np.all(np.abs(x) < 1e-6)


def test__rosenbrock_problem__converges(s):
    x0 = np.array([3.14, 2.72, 42.0, 9.81, 8.31])

    s.set_initial_guess(x0)
    s.set_objective_fn(rosenbrock_objective_fn)

    status, message = s.minimize()
    x = s.x

    assert status == 0
    assert len(x) == len(x0)
    assert np.all(np.abs(x - 1.0) < 1e-5)  # The solution is [1, 1, ..., 1].

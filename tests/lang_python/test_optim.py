import numpy as np
import pytest
from openinterfaces.interfaces.optim import Optim


# -----------------------------------------------------------------------------
# Problems
def objective_fn(x):
    return np.sum(x**2)


@pytest.fixture(params=["scipy_optimize"])
def s(request):
    return Optim(request.param)


# -----------------------------------------------------------------------------
# Tests
def test__simple_convex_problem__converges(s):
    x0 = np.array([0.5, 0.6, 0.7])

    s.set_initial_guess(x0)
    s.set_objective_fn(objective_fn)

    result = s.minimize()

    assert result.status == 0
    assert np.all(np.abs(result.x) < 1e-6)

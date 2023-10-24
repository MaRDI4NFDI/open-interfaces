import numpy as np
import numpy.testing as npt
import pytest
from oif.interfaces.linear_solver import LinearSolver


class TestLinearSolverViaNumPyImplementation:
    @pytest.fixture(params=["c_lapack", "numpy"])
    def s(self, request):
        return LinearSolver(request.param)

    def test_1(self, s):
        A = np.array(
            [
                [1.0, 1.0],
                [-3.0, 1.0],
            ]
        )
        b = np.array([6.0, 2.0])
        x = s.solve(A, b)

        npt.assert_allclose(A @ x, b, rtol=1e-15, atol=1e-15)

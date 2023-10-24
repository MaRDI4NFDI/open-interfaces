import numpy as np
import numpy.testing as npt
import pytest
from oif.interfaces.qeq_solver import QeqSolver


class TestQeqSolverViaPyQeqSolverImplementation:
    @pytest.fixture()
    def s(self):
        return QeqSolver("py_qeq_solver")

    def test_1(self, s):
        a, b, c = 1.0, 5.0, 4.0
        x = s.solve(a, b, c)

        npt.assert_allclose(x, np.array([-4.0, -1.0]), rtol=1e-15)

    def test_2(self, s):
        a, b, c = 1.0, -2.0, 1.0
        x = s.solve(a, b, c)

        npt.assert_allclose(x, np.array([1.0, 1.0]), rtol=1e-15)

    def test_3(self, s):
        a, b, c = 1.0, -2.0, 1.0
        x = s.solve(a, b, c)

        npt.assert_allclose(x, np.array([1.0, 1.0]), rtol=1e-15)

    def test_qeq_extreme_roots_should_give_several_digits(self, s):
        a, b, c = 1.0, -20_000.0, 1.0
        x = s.solve(a, b, c)
        x.sort()

        npt.assert_allclose(
            x, np.array([5.000000055588316e-05, 19999.999949999998]), rtol=1e-6
        )

    def test_correct_extreme_roots__should_give_fiveteen_digits(self, s):
        a, b, c = 1.0, -20_000.0, 1.0
        x = s.solve(a, b, c)
        x.sort()

        npt.assert_allclose(
            x,
            np.array([5.000000012500001e-05, 19999.999949999998]),
            rtol=1e-15,
            atol=1e-15,
        )

import numpy as np
from oif.util import laplacian2DMatrix
from scipy import sparse as sp


def test_laplacian__constant_function__should_be_zero():
    N = 5

    left = -1.0
    right = 1.0
    top = 1.0
    bottom = -1.0
    dx = (right - left) / N
    dy = (top - bottom) / N
    X, Y = np.meshgrid(
        np.linspace(left, right, num=N + 1), np.linspace(top, bottom, num=N + 1)
    )
    F = X * Y
    F[:, :] = 1.0

    Lap = laplacian2DMatrix(N + 1, dx, dy)
    LaplacianF = Lap.dot(np.reshape(F, (-1,)))
    LaplacianF_2D = np.reshape(LaplacianF, (N + 1, N + 1))
    LaplacianF_2D_inner = LaplacianF_2D[1:-1, 1:-1]
    np.testing.assert_allclose(LaplacianF_2D_inner, 0.0, rtol=1e-12, atol=1e-12)


def test_laplacian__linear_function__should_be_zero():
    N = 10

    left = -1.0
    right = 1.0
    top = 1.0
    bottom = -1.0
    dx = (right - left) / N
    dy = (top - bottom) / N
    X, Y = np.meshgrid(
        np.linspace(left, right, num=N + 1), np.linspace(top, bottom, num=N + 1)
    )
    F = 21 * X + 5 * Y

    Lap = laplacian2DMatrix(N + 1, dx, dy)
    LaplacianF = Lap.dot(np.reshape(F, (-1,)))
    LaplacianF_2D = np.reshape(LaplacianF, (N + 1, N + 1))
    LaplacianF_2D_inner = LaplacianF_2D[1:-1, 1:-1]
    np.testing.assert_allclose(LaplacianF_2D_inner, 0.0, rtol=1e-12, atol=1e-12)


def test_laplacian_with_harmonic_function():
    N = 400

    left = -1.0
    right = 1.0
    top = 0.5
    bottom = -0.5
    dx = (right - left) / N
    dy = (top - bottom) / N
    X, Y = np.meshgrid(
        np.linspace(left, right, num=N + 1), np.linspace(top, bottom, num=N + 1)
    )
    F = np.exp(X) * np.cos(Y)

    Lap = laplacian2DMatrix(N + 1, dx, dy)
    LaplacianF = Lap.dot(np.reshape(F, (-1,)))
    LaplacianF_2D = np.reshape(LaplacianF, (N + 1, N + 1))
    LaplacianF_2D_inner = LaplacianF_2D[1:-1, 1:-1]
    np.testing.assert_allclose(LaplacianF_2D_inner, 0.0, rtol=1e-5, atol=1e-5)


def test_laplacian__solve_poisson_equation():
    # Define the RHS function:
    def source(x, y):
        return -(20 * y**3 + 9 * np.pi**2 * (y - y**5.0)) * np.sin(3 * np.pi * x)

    # Set up the grid:
    m = 101
    x = np.linspace(0, 1, m)
    x = x[1:-1]
    y = np.linspace(0, 1, m)
    y = y[1:-1]
    X, Y = np.meshgrid(x, y)
    # Set up and solve the linear system
    A = laplacian2DMatrix(m - 2, 1 / (m - 1), 1 / (m - 1))
    F = source(X, Y).reshape([(m - 2) ** 2])
    U = sp.linalg.spsolve(A, F)
    U_2D = np.reshape(U, (m - 2, m - 2))
    U_exact = (Y - Y**5) * np.sin(3 * np.pi * X)

    np.testing.assert_allclose(U_2D, U_exact, rtol=5e-4, atol=5e-4)

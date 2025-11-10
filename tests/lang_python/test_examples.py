import subprocess

import pytest


@pytest.mark.parametrize(
    "impl_name", ["c_qeq_solver", "py_qeq_solver", "jl_qeq_solver"]
)
def test_call_qeq_from_python__exit_success(impl_name: str):
    p = subprocess.run(["python", "examples/call_qeq_from_python.py", impl_name])

    assert p.returncode == 0


@pytest.mark.parametrize(
    "impl_name",
    ["c_lapack", "jl_backslash", "numpy"],
)
def test_call_linsolve_from_python__exit_success(impl_name: str):
    p = subprocess.run(["python", "examples/call_linsolve_from_python.py", impl_name])

    assert p.returncode == 0


@pytest.mark.parametrize(
    "impl_name",
    ["sundials_cvode", "jl_diffeq", "scipy_ode"],
)
def test_call_ivp_from_python__exit_success(impl_name: str):
    p = subprocess.run(["python", "examples/call_ivp_from_python.py", impl_name])

    assert p.returncode == 0


# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "impl_name",
    ["sundials_cvode", "jl_diffeq", "scipy_ode"],
)
def test_call_ivp_from_python_burgers_eq__exit_success(impl_name: str):
    p = subprocess.run(
        [
            "python",
            "examples/call_ivp_from_python_burgers_eq.py",
            "--no-plot",
            impl_name,
        ]
    )

    assert p.returncode == 0


# -----------------------------------------------------------------------------
# def test_call_ivp_from_python_vdp__dopri5__fails(impl_name: str):

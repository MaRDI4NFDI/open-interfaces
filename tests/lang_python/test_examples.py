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
# Van der Pol equation: unsuccessful and successful runs.


@pytest.mark.parametrize(
    "unsuccessful_impl",
    [
        "scipy_ode-dopri5",
        "scipy_ode-dopri5-100k",
        "scipy_ode-vode",
    ],
)
def test_call_ivp_from_python_vdp__dopri5__fails(unsuccessful_impl: str):
    p = subprocess.run(
        ["python", "examples/call_ivp_from_python_vdp.py", unsuccessful_impl],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    assert "RuntimeError: Error occurred while executing method 'integrate'" in p.stderr


@pytest.mark.parametrize(
    "successful_impl",
    [
        "scipy_ode-vode-40k",
        # "sundials_cvode-default",  # It is simply too slow to run (40 secs)
        "jl_diffeq-rosenbrock23",
    ],
)
def test_call_ivp_from_python_vdp__with_successful_impl_succeeds(successful_impl: str):
    p = subprocess.run(
        [
            "python",
            "examples/call_ivp_from_python_vdp.py",
            "--do-not-save-anything",
            successful_impl,
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0

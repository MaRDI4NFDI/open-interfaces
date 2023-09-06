import argparse

from oif.frontend_python.qeq_solver import QeqSolver


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["c", "python"], nargs="?")
    args = p.parse_args()
    if args.backend is None:
        args.backend = "c"
    return args


def main():
    args = _parse_args()
    backend = args.backend
    print("Calling from Python an open interface for quadratic solver")
    print(f"Backend: {backend}")
    s = QeqSolver(backend)
    a, b, c = 1.0, 5.0, 4.0
    x = s.solve(a, b, c)

    print(f"Solving quadratic equation for a = {a}, b = {b}, c = {c}")
    print(f"x = {x}")


if __name__ == "__main__":
    main()

import argparse
import dataclasses

from oif.interfaces.qeq import QEQ


@dataclasses.dataclass
class Args:
    impl: str


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "impl",
        choices=["c_qeq_solver", "py_qeq_solver", "jl_qeq_solver"],
        default="c_qeq_solver",
        nargs="?",
    )
    args = p.parse_args()
    return Args(**vars(args))


def main():
    args = _parse_args()
    impl = args.impl
    print("Calling from Python an open interface for quadratic solver")
    print(f"Implementation: {impl}")
    s = QEQ(impl)
    a, b, c = 1.0, 5.0, 4.0
    x = s.solve(a, b, c)

    print(f"Solving quadratic equation for a = {a}, b = {b}, c = {c}")
    print(f"x = {x}")


if __name__ == "__main__":
    main()

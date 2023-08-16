from oif.qeq_solver import QeqSolver


def main():
    s = QeqSolver("c")
    a, b, c = 1.0, 5.0, 4.0
    x = s.solve(a, b, c)

    print(f"Solving quadratic equation for a = {a}, b = {b}, c = {c}")
    print(f"x = {x}")


if __name__ == "__main__":
    main()

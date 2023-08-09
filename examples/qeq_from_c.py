from oif.qeq_solver import QeqSolver

s = QeqSolver("c")
a, b, c = 1.0, 2.0, 1.0
x = s.solve(a, b, c)

print(f"Solving quadratic equation for a = {a}, b = {b}, c = {c}")
print(f"x = {x}")

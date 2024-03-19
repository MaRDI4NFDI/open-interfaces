import numpy as np
from oif.interfaces.ivp import IVP
from scikits.odes.ode import ode


def rhs(t, y, ydot):
    ydot[:] = -y


s_oif = IVP("sundials_cvode")
s = ode(
    "cvode",
    rhs,
    old_api=False,
    lmm_type="ADAMS",
    nonlinsolver="fixedpoint",
    rtol=1e-15,
    atol=1e-15,
)

y0 = np.array([1.0])
t0 = 0.0

s_oif.set_initial_value(y0, t0)
s.init_step(t0, y0)
s_oif.set_rhs_fn(rhs)

for t in [0.1, 0.2]:
    s_oif.integrate(t)
    output = s.step(t)
    print("s_oif.y: ", s_oif.y)
    print("sundials.y: ", output.values.y)

s_oif.print_stats()
# print(s.get_info())
print("=========================================================")
print()
s.print_stats()

ivp
===

.. py:module:: ivp

.. autoapi-nested-parse::

   This module defines the interface for solving initial-value problems
   for ordinary differential equations:

       .. math::
           \frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0.







Module Contents
---------------

.. py:data:: RhsFn
   :type:  TypeAlias
   :value: Callable[[float, np.ndarray, np.ndarray, object], int]


   Signature of the right-hand side (RHS) function :math:`f(t, y)`.

   The function accepts four arguments:
       - `t`: current time,
       - `y`: state vector at time :math:`t`,
       - `ydot`: output array to which the result of function evalutation is stored,
       - `user_data`: additional context (user-defined data) that
         must be passed to the function (e.g., parameters of the system).

.. py:class:: IVP(impl: str)

   Interface for solving initial value problems.

   This class serves as a gateway to the implementations of the
   solvers for initial-value problems for ordinary differential equations.

   :param impl: Name of the desired implementation.
   :type impl: str

   .. rubric:: Examples

   Let's solve the following initial value problem:

   .. math::
       y'(t) = -y(t), \quad y(0) = 1.

   First, import the necessary modules:
   >>> import numpy as np
   >>> from oif.interfaces.ivp import IVP

   Define the right-hand side function:

   >>> def rhs(t, y, ydot, user_data):
   ...     ydot[0] = -y[0]
   ...     return 0  # No errors, optional

   Now define the initial condition:

   >>> y0, t0 = np.array([1.0]), 0.0

   Create an instance of the IVP solver using the implementation "jl_diffeq",
   which is an adapter to the `OrdinaryDiffeq.jl` Julia package:

   >>> s = IVP("jl_diffeq")

   We set the initial value, the right-hand side function, and the tolerance:

   >>> s.set_initial_value(y0, t0)
   >>> s.set_rhs_fn(rhs)
   >>> s.set_tolerances(1e-6, 1e-12)

   Now we integrate to time `t = 1.0` in a loop, outputting the current value
   of `y` with time step `0.1`:

   >>> t = t0
   >>> times = np.linspace(t0, t0 + 1.0, num=11)
   >>> for t in times[1:]:
   ...     s.integrate(t)
   ...     print(f"{t:.1f} {s.y[0]:.6f}")
   0.1 0.904837
   0.2 0.818731
   0.3 0.740818
   0.4 0.670320
   0.5 0.606531
   0.6 0.548812
   0.7 0.496585
   0.8 0.449329
   0.9 0.406570
   1.0 0.367879


   .. py:attribute:: y
      :type:  numpy.ndarray

      Current value of the state vector.


   .. py:method:: set_initial_value(y0: numpy.ndarray, t0: float)

      Set initial value y(t0) = y0.



   .. py:method:: set_rhs_fn(rhs_fn: RhsFn)

      Specify right-hand side function f.



   .. py:method:: set_tolerances(rtol: float, atol: float)

      Specify relative and absolute tolerances, respectively.



   .. py:method:: set_user_data(user_data: object)

      Specify additional data that will be used for right-hand side function.



   .. py:method:: set_integrator(integrator_name: str, integrator_params: dict = {})

      Set integrator, if the name is recognizable.



   .. py:method:: integrate(t: float)

      Integrate to time `t` and write solution to `y`.



   .. py:method:: print_stats()

      Print integration statistics.

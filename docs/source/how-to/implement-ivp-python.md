# How to implement `IVP` interface in Python

Let's have a look at how to implement the `IVP` interface in Python.

We will use the [classical Runge--Kutta 4th order method (RK4)][rk4].
In this method, given the solution $y_n$ at time $t_n$ and time step $h$,
the solution $y_{n+1}$
at the next time step $t_{n+1} = t_n + h$ is computed as
```{math}
:label: ivp-py-rk4-next-step
y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4),
```
where
```{math}
:label: ivp-py-rk4-aux
k_1 &= f \left( t_n, y_n \right), \\
k_2 &= f \left( t_n + 0.5h, y_n + 0.5h k_1 \right), \\
k_3 &= f \left( t_n + 0.5h, y_n + 0.5h k_2 \right), \\
k_4 &= f \left( t_n + h, y_n + hk_3 \right).
```

For simplicity, we implement the RK4 method only with fixed time step
(so, it is user's responsibility to choose the time step small enough).

The implementation should inherit from the abstract base class
[`IVPInterface`][ivp-interface],
which has the following abstract methods:
```python
class IVPInterface(abc.ABC):
    @abc.abstractmethod
    def set_initial_value(self, y0: np.ndarray, t0: float) -> Union[int, None]:
        """Set initial value y(t0) = y0."""

    @abc.abstractmethod
    def set_rhs_fn(self, rhs: Callable) -> Union[int, None]:
        """Specify right-hand side function f."""

    @abc.abstractmethod
    def set_tolerances(self, rtol: float, atol: float) -> Union[int, None]:
        """Specify relative and absolute tolerances, respectively."""

    @abc.abstractmethod
    def set_user_data(self, user_data: object) -> Union[int, None]:
        """Specify additional data that will be used for right-hand side function."""

    @abc.abstractmethod
    def set_integrator(
        self, integrator_name: str, integrator_params: Dict
    ) -> Union[int, None]:
        """Set integrator, if the name is recognizable."""

    @abc.abstractmethod
    def integrate(self, t: float, y: np.ndarray) -> Union[int, None]:
        """Integrate to time `t` and write solution to `y`."""

    @abc.abstractmethod
    def print_stats(self):
        """Print integration statistics."""
```

We start by creating a new Python file named `rk4.py`
and importing the `IVPInterface` class and NumPy:
```python
import numpy as np
from openinterfaces._impl.interfaces.ivp import IVPInterface

from typing import Callable, Dict, Union
```
Note that we also import `Callable`, `Dict`, and `Union` from the `typing` module
to use them in type hints (which is completely optional, of course, in Python).

Then we declare our implementation class as a subclass of `IVPInterface`:
and define the `__init__` method to initialize the instance variables
(most of them are just type hints here, because we will initialize them later
in the respective methods, however, `user_data` and `n_rhs_evals` are
initialized here because they are not necessarily initialized later):
```python
class RK4FixedStepIntegrator(IVPInterface):
    def __init__(self):
        self.y: np.ndarray
        self.t: float
        self.rhs: Callable
        self.user_data = None
        self.n_rhs_evals = 0  # Number of right-hand side evaluations
        # Auxiliary arrays for RK4
        self.k1: np.ndarray
        self.k2: np.ndarray
        self.k3: np.ndarray
        self.k4: np.ndarray
```

Next, we need to store the initial value and time in the `set_initial_value` method:
```python
    def set_initial_value(self, y0: np.ndarray, t0: float) -> Union[int, None]:
        self.y = y0
        self.t = t0
        self.k1 = np.zeros_like(y0)
        self.k2 = np.zeros_like(y0)
        self.k3 = np.zeros_like(y0)
        self.k4 = np.zeros_like(y0)
```
where we also allocate auxiliary arrays for the RK4 method.
Then we set the right-hand side function in the `set_rhs_fn` method:
```python
    def set_rhs_fn(self, rhs: Callable) -> Union[int, None]:
        self.rhs = rhs
```
and any user data that the user wants to pass to the right-hand side function
in the `set_user_data` method:
```python
    def set_user_data(self, user_data: object) -> Union[int, None]:
        self.user_data = user_data
```

Because we are using fixed time step and our implementation implements
only one algorithm (integrator), we can keep the `set_tolerances` and
`set_integrator` methods empty:
```python
    def set_tolerances(self, rtol: float, atol: float) -> Union[int, None]:
        pass

    def set_integrator(
        self, integrator_name: str, integrator_params: Dict
    ) -> Union[int, None]:
        pass
```
Otherwise, we would need to store the tolerances and integrator parameters
as properties (as we do with the initial value, right-hand side function,
and user data, and reinitialize the integrator accordingly if needed.

At last, we implement the `integrate` method using the
{eq}`ivp-py-rk4-next-step` and {eq}`ivp-py-rk4-aux`:
```python
    def integrate(self, t: float, y: np.ndarray) -> Union[int, None]:
        h = t - self.t  # Time step
        self.rhs(self.t, self.y, self.k1[:], self.user_data)
        self.rhs(self.t + h / 2, self.y + h / 2 * self.k1, self.k2[:], self.user_data)
        self.rhs(self.t + h / 2, self.y + h / 2 * self.k2, self.k3[:], self.user_data)
        self.rhs(self.t + h, self.y + h * self.k3, self.k4[:], self.user_data)
        self.n_rhs_evals += 4  # We evaluated the right-hand side four times
        y[:] = self.y + h / 6 * (self.k1 + 2 * self.k2 + 2 * self.k3 + self.k4)
        self.t = t
        self.y = y  # Write the solution to the output array
```
Note that we also update the number of right-hand side evaluations
in each call to the `integrate` method.
Besides, we write into the output array `y` the computed solution
and keep a link to it in `self.y` for the next call to `integrate`.
Additionally, we use internal (implementation-dependent) arrays `k1`, `k2`,
`k3`, and `k4` to store the auxiliary values defined by the logic
of the RK4 method.

Finally, we implement the `print_stats` method to print the number of
right-hand side evaluations:
```python
    def print_stats(self):
        print(f"Number of right-hand side evaluations: {self.n_rhs_evals}")
```

## Make the implementation available

We want to make our implementation discoverable by _Open Interfaces_.
To do that, we need to do two things:
- Choose the name for the implementation (here, we choose `rk4_example`)
- Put the implementation in a folder hierarchy somewhere on disk:
  for example, `$HOME/oif_impl/ivp/our_rk4/RK4.py`,
- Create a file named `rk4_example.conf`
  (its basename is the same as the implementation name)
  in the `$HOME/oif_impl/ivp/rk4_example/` with the following content:
  ```
  python
  ivp.rk4_example.rk4 RK4FixedStepIntegrator
  ```
  In this file, the first line specifies the language of the implementation
  while the second line specifies the module to import
  and the class name that will be instantiated by _Open Interfaces_.
- Set the environment variable `OIF_IMPL_PATH` to point to `$HOME/oif_impl`:
  ```shell
  export OIF_IMPL_PATH=$HOME/oif_impl:$OIF_IMPL_PATH
  ```
- Because we put our implementation in a non-standard location,
  we also need to set the `PYTHONPATH` environment variable
  to include `$HOME` (so that Python can find our implementation):
  ```shell
  export PYTHONPATH=$HOME/oif_impl:$PYTHONPATH
  ```
  so that `import ivp.rk4_example.rk4` works correctly (it will be done
  automatically by _Open Interfaces_ for us).

Now we are ready to use our implementation.

## Usage example

To test how our implementation works,
we can solve a simple ODE
$y' = -y$, $y(0) = 1$ with the exact solution $y(t) = e^{-t}$.

We start with imports:
```python
import numpy as np
from openinterfaces.interfaces.ivp import IVP
```

Then, we define the right-hand side function:
```python
def rhs(t, y, dydt, user_data):
    dydt[:] = -y
    return 0
```
where `return 0` indicates success.

Then we instantiate the Gateway component `IVP`
and pass the name of the desired implementation to it:
```python
solver = IVP("rk4_example")
```
and set the initial value,
right-hand side function, and user data (if any):
```python
solver.set_initial_value(np.array([1.0]), 0.0)
solver.set_rhs_fn(rhs)
```

And then we can integrate to a desired time using relatively small time steps
(recall that our implementation uses fixed time step)
and print the numerical and exact solutions at each time point:
```python
times = np.linspace(0, 1, 11)
for t in times[1:]:
    y = np.zeros(1)
    solver.integrate(t, y)
    print(f"t = {t:.2f}, y = {y[0]:.4f}, exact = {np.exp(-t):.4f}")
```

Finally, we can print the statistics of the integration:
```python
solver.print_stats()
```

The complete code of the usage example is as follows:
```python
import numpy as np
from openinterfaces.interfaces.ivp import IVP


def rhs(t, y, dydt, user_data):
    dydt[:] = -y
    return 0


def main():
    solver = IVP("rk4_example")
    solver.set_initial_value(np.array([1.0]), 0.0)
    solver.set_rhs_fn(rhs)
    solver.set_user_data(None)

    times = np.linspace(0, 1, 11)
    for t in times[1:]:
        solver.integrate(t)
        print(f"t = {t:.2f}, y = {solver.y[0]:.4f}, exact = {np.exp(-t):.4f}")

    solver.print_stats()


if __name__ == "__main__":
    main()
```
with output:
```shell
[dispatch] Configuration file: /home/user/oif_impl/ivp/rk4_example/rk4_example.conf
[dispatch] Backend name: python
[dispatch] Implementation details: 'ivp.rk4_example.rk4fixedstep
RK4FixedStepIntegrator'
[bridge_python] Backend is already initialized
[bridge_python] libpython path: /home/user/sw/miniforge3/envs/um02-open-interfaces/lib
[bridge_python] Loading libpython3.13.so
[dispatch_python] /home/user/sw/miniforge3/envs/um02-open-interfaces/bin/python
[dispatch_python] 3.13.2 | packaged by conda-forge | (main, Feb 17 2025, 14:10:22) [GCC 13.3.0]
[dispatch_python] NumPy version:  2.2.4
[bridge_python] Provided module name: 'ivp.rk4_example.rk4fixedstep'
[bridge_python] Provided class name: 'RK4FixedStepIntegrator'
t = 0.10, y = 0.9048, exact = 0.9048
t = 0.20, y = 0.8187, exact = 0.8187
t = 0.30, y = 0.7408, exact = 0.7408
t = 0.40, y = 0.6703, exact = 0.6703
t = 0.50, y = 0.6065, exact = 0.6065
t = 0.60, y = 0.5488, exact = 0.5488
t = 0.70, y = 0.4966, exact = 0.4966
t = 0.80, y = 0.4493, exact = 0.4493
t = 0.90, y = 0.4066, exact = 0.4066
t = 1.00, y = 0.3679, exact = 0.3679
Number of right-hand side evaluations: 40
```


[rk4]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
[ivp-interface]: https://github.com/MaRDI4NFDI/open-interfaces/blob/main/lang_python/oif_impl/openinterfaces/_impl/interfaces/ivp.py

# Comparison of performance between using `scipy.integrate.ode` versus `ivp` interface

Here we provide comparison of performance differences between "native"
approach, when user code uses numerical packages provided by their language
of choice, and open-interfaces approach, when the user code interacts
with numerical packages via `liboif`.

To conduct the comparison, we solve the initial-value problem
for the inviscid Burgers' equation with periodic boundary conditions:
```{math}
:label: problem

\begin{align}
    u_t + \left(\frac{u^2}{2}\right)_x &= 0,
        \quad x \in [0, 2], \quad t \in [0, 2], \\
    u(0, x) &= 0.5 - 0.25 \sin(\pi x), \\
    u(t, 0) &= u(t, 2)
\end{align}
```
using the method-of-lines approach.
In this approach, one converts a partial differential equation to a system
of ordinary differential equations that can be integrated by third-party
solvers for initial-value problems (time integrators).

We use the following three different implementations for time integration:
- via `ivp` interface using `scipy_ode_dopri5` implementation
- via `ivp` interface using `sundials_cvode` implementation
- via direct use of `scipy.integrate.ode` package with `dopri5` method
  (called `native_scipy_dopri5` in the results below)

Note that the first and the last implementations in the list above are the
same (Runge--Kutta method of 5th order with embedded
4th order method for error estimation by Dormand & Prince).
Besides, `dopri5` method from the `scipy.integrate.ode` package is actually
a wrapper over Fortran implementation, so it is not purely native.

To ensure statistically meaningful results, we integrate with each
implementation multiple times and report the average time-to-solution
as well as standard deviation.
Besides that, we analyze the scalability, that is, how performance
changes with the increase of the problem size (in this case, grid resolution
$N$).

As the problem {eq}`problem` is based on a hyperbolic PDE,
increase of the grid resolution $N$ means proportional decrease
of the maximum allowed time step for numerical stability,
which means that with the increase of the number of grid points
the number of invocations of integration functions increases as well.

To produce the results below, we run the following command:
```shell
python examples/compare_performance_ivp_burgers_eq.py all --n_runs 10
```
where `--n_runs` specifies number of runs for obtaining statistics for each
implementation and grid resolution.

Note that all implementations used absolute and relative tolerances
set to $10^{-15}$.

The results:
```
Statistics:
N = 101
scipy_ode_dopri5           0.42   0.01
sundials_cvode             0.46   0.01
native_scipy_dopri5        0.41   0.01
N = 1001
scipy_ode_dopri5           2.93   0.07
sundials_cvode             4.28   0.12
native_scipy_dopri5        2.77   0.03
N = 10001
scipy_ode_dopri5          78.06   2.61
sundials_cvode           122.70   1.34
native_scipy_dopri5       72.16   0.64
```

This is a debug variant, in which Python callable is replaced by a C function
for `scipy_ode_dopri5`:
```
Statistics:
N = 101
scipy_ode_dopri5           1.40   0.10
sundials_cvode             0.46   0.01
native_scipy_dopri5        0.39   0.02
N = 1001
scipy_ode_dopri5           7.39   0.09
sundials_cvode             4.28   0.05
native_scipy_dopri5        2.75   0.07
N = 10001
scipy_ode_dopri5         107.32   1.95
sundials_cvode           126.12   1.65
```

## 2024-01-31 Profiling conversion between NumPy ndarrays and OIFArrayF64 data structures

Using the script `examples/compare_performance_ivp_burgers_eq_pure_comparison.py one scipy_ode_dopri5`, we get the following the following results:

Profile the code with the following command:
```
python -m cProfile -o profiler-results \
    examples/compare_performance_ivp_burgers_eq_pure_comparison.py \
        one scipy_ode_dopri5
```

than using
```
python -m pstats profiler-results
sort cumulative
stats 20
```

we obtain the following results:
```
Wed Jan 31 16:54:54 2024    profiler-results

         4223975 function calls (4196427 primitive calls) in 8.631 seconds

   Ordered by: cumulative time
   List reduced from 5448 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    744/1    0.011    0.000    8.632    8.632 {built-in method builtins.exec}
        1    0.000    0.000    8.632    8.632 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:1(<module>)
        1    0.000    0.000    7.706    7.706 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:177(run_one_impl)
        1    0.005    0.005    7.706    7.706 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:252(_run_once)
     2000    0.005    0.000    7.378    0.004 /home/dima/sw/mambaforge/envs/um02-toy-example/lib/python3.12/site-packages/scipy/integrate/_ode.py:397(integrate)
     2000    0.497    0.000    7.373    0.004 /home/dima/sw/mambaforge/envs/um02-toy-example/lib/python3.12/site-packages/scipy/integrate/_ode.py:1173(run)
    89871    0.268    0.000    6.875    0.000 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:125(compute)
    89871    0.087    0.000    3.945    0.000 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:112(compute_rhs_native)
    89871    3.024    0.000    3.858    0.000 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:94(compute_rhs)
    89871    1.218    0.000    1.874    0.000 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:134(_c_args_from_py_args)
   826/10    0.005    0.000    1.034    0.103 <frozen importlib._bootstrap>:1349(_find_and_load)
   821/10    0.004    0.000    1.033    0.103 <frozen importlib._bootstrap>:1304(_find_and_load_unlocked)
   784/11    0.003    0.000    1.031    0.094 <frozen importlib._bootstrap>:911(_load_unlocked)
   671/11    0.002    0.000    1.031    0.094 <frozen importlib._bootstrap_external>:988(exec_module)
  1958/21    0.002    0.000    1.028    0.049 <frozen importlib._bootstrap>:480(_call_with_frames_removed)
  1006/97    0.002    0.000    0.911    0.009 <frozen importlib._bootstrap>:1390(_handle_fromlist)
   659/71    0.001    0.000    0.906    0.013 {built-in method builtins.__import__}
    90031    0.123    0.000    0.814    0.000 /home/dima/sw/mambaforge/envs/um02-toy-example/lib/python3.12/site-packages/numpy/core/fromnumeric.py:2692(max)
    89871    0.376    0.000    0.753    0.000 examples/compare_performance_ivp_burgers_eq_pure_comparison.py:154(_py_args_from_c_args)
    90230    0.210    0.000    0.692    0.000 /home/dima/sw/mambaforge/envs/um02-toy-example/lib/python3.12/site-packages/numpy/core/fromnumeric.py:71(_wrapreduction)
```

where we can see that inside the function `compute` that runs for 6.875
seconds,
the functions `_c_args_from_py_args` and `_py_args_from_c_args` take
$1.874 + 0.753 = 2.627$ seconds, which constitutes 38 % of the run time
for the function `compute` (useful work is done inside the function
`compute_rhs_native` during the other 62 % of run time).

Line profiling was done with
```shell
    kernprof -l -v examples/compare_performance_ivp_burgers_eq_pure_comparison.py one scipy_ode_dopri5
```
with functions `_c_args_from_py_args` and `_py_args_from_c_args` annotated
(decorator `@profile` from `line_profiler` module).

The results are:
```
================================================================
Solving Burgers' equation with time integration scipy_ode_dopri5
Finished
Wrote profile results to compare_performance_ivp_burgers_eq_pure_comparison.py.lprof
Timer unit: 1e-06 s

Total time: 2.43264 s
File: examples/compare_performance_ivp_burgers_eq_pure_comparison.py
Function: _c_args_from_py_args at line 135

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   135                                               @profile
   136                                               def _c_args_from_py_args(self, *args) -> list:
   137     89871      56466.2      0.6      2.3          begin = time.time()
   138     89871      33840.0      0.4      1.4          c_args: list[CtypesType] = []
   139    269613     245270.1      0.9     10.1          for i, (t, v) in enumerate(zip(self.arg_types, args)):
   140    179742      55548.1      0.3      2.3              if t == OIF_INT:
   141                                                           c_args.append(ctypes.c_int(v))
   142    179742      52412.0      0.3      2.2              elif t == OIF_FLOAT64:
   143     89871      76238.6      0.8      3.1                  c_args.append(ctypes.c_double(v))
   144     89871      28563.6      0.3      1.2              elif t == OIF_ARRAY_F64:
   145     89871     118409.0      1.3      4.9                  assert v.dtype == np.float64
   146     89871      29576.1      0.3      1.2                  nd = v.ndim
   147     89871     194035.4      2.2      8.0                  dimensions = (ctypes.c_long * len(v.shape))(*v.shape)
   148     89871      42022.9      0.5      1.7                  double_p_t = ctypes.POINTER(ctypes.c_double)
   149     89871    1001718.9     11.1     41.2                  data = v.ctypes.data_as(double_p_t)
   150
   151     89871     270073.7      3.0     11.1                  oif_array = OIFArrayF64(nd, dimensions, data)
   152     89871     109151.4      1.2      4.5                  c_args.append(ctypes.pointer(oif_array))
   153     89871      37293.8      0.4      1.5          end = time.time()
   154                                                   # print("{:35s} {:.2e}".format("Elapsed time _c_args_from_py_args", end - begin))
   155     89871      82017.4      0.9      3.4          return c_args

Total time: 0.901177 s
File: examples/compare_performance_ivp_burgers_eq_pure_comparison.py
Function: _py_args_from_c_args at line 157

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   157                                               @profile
   158                                               def _py_args_from_c_args(self, *args) -> list:
   159     89871      39935.7      0.4      4.4          begin = time.time()
   160     89871      24564.8      0.3      2.7          py_args = []
   161     89871      32901.2      0.4      3.7          py_args.append(args[0])
   162     89871      18304.0      0.2      2.0          v = args[1]
   163     89871     150222.5      1.7     16.7          arr_type = ctypes.c_double * v.contents.dimensions[0]
   164    179742      52523.6      0.3      5.8          py_args.append(
   165    179742     395453.0      2.2     43.9              np.ctypeslib.as_array(
   166     89871     107230.1      1.2     11.9                  arr_type.from_address(ctypes.addressof(v.contents.data.contents))
   167                                                       )
   168                                                   )
   175     89871      28710.8      0.3      3.2          end = time.time()
   176                                                   # print("{:35s} {:.2e}".format("Elapsed time _py_args_from_c_args", end - begin))
   177
   178     89871      51331.4      0.6      5.7          return py_args
```

We can see from the results of line profiling, that somehow functions
`ndarray.ctypes.data_as` and `numpy.ctypeslib.as_array` are taking 40 % of run
time, and look as the main targets for optimization.

# 2024-02-08 Performance assessment of conversion Python arguments to C + callback

I have implemented invocation of a C callback along with conversion of Python
arguments to C types as a Python C extension.

In details in the following purely Python code:
```python
import ctypes
import time

import _callback
import numpy as np
from oif.core import OIF_ARRAY_F64, OIF_FLOAT64, OIF_INT, OIFArrayF64


class Callback:
    def __init__(self, fn_p, id: str):
        # Details of creating a callback function from a PyCapsule `fn_p`.
        # Omitted as irrevelant here.

    def __call__(self, *args):
        c_args = []
        for i, (t, v) in enumerate(zip(self.arg_types, args)):
            if t == OIF_INT:
                c_args.append(ctypes.c_int(v))
            elif t == OIF_FLOAT64:
                c_args.append(ctypes.c_double(v))
            elif t == OIF_ARRAY_F64:
                assert v.dtype == np.float64
                nd = v.ndim
                dimensions = (ctypes.c_long * len(v.shape))(*v.shape)
                data = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                oif_array = OIFArrayF64(nd, dimensions, data)
                c_args.append(ctypes.pointer(oif_array))

        return self.fn_p_py(*c_args)
```

the method `__call__` is update to use an equivalent C code.

Here I report the performance results.

With the previous code, `cProfile` gives the following results:
```
Thu Feb  8 15:34:35 2024    profiler-results-oif-scipy_ode_dopri5

         5028526 function calls (5001575 primitive calls) in 10.659 seconds

   Ordered by: cumulative time
   List reduced from 5549 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    758/1    0.009    0.000   10.660   10.660 {built-in method builtins.exec}
        1    0.000    0.000   10.660   10.660 examples/compare_performance_ivp_burgers_eq.py:1(<module>)
        1    0.000    0.000    9.822    9.822 examples/compare_performance_ivp_burgers_eq.py:96(run_one_impl)
        1    0.003    0.003    9.822    9.822 examples/compare_performance_ivp_burgers_eq.py:194(_run_once)
     2000    0.004    0.000    9.405    0.005 oif/interfaces/python/oif/interfaces/ivp.py:38(integrate)
     2002    0.064    0.000    9.403    0.005 oif/lang_python/oif/core.py:138(call)
     2000    0.006    0.000    9.275    0.005 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:39(integrate)
     2000    0.004    0.000    9.268    0.005 <>/lib/python3.12/site-packages/scipy/integrate/_ode.py:397(integrate)
     2000    0.394    0.000    9.263    0.005 <>/lib/python3.12/site-packages/scipy/integrate/_ode.py:1173(run)
    89872    0.197    0.000    8.869    0.000 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:44(_rhs_fn_wrapper)
    89872    2.109    0.000    8.672    0.000 oif_impl/python/oif/callback.py:27(__call__)
    89872    0.645    0.000    5.703    0.000 oif/lang_python/oif/core.py:105(wrapper)
    89872    3.066    0.000    3.914    0.000 examples/compare_performance_ivp_burgers_eq.py:73(compute_rhs)
   179744    0.311    0.000    1.090    0.000 <>/lib/python3.12/site-packages/numpy/ctypeslib.py:506(as_array)
```

we can see that the call invocation took 2.109 seconds by itself
(that is, for conversion of the arguments to C types from Python types).

With the new implementation of the `__call__` method:
```
Thu Feb  8 15:25:22 2024    profiler-results-oif-scipy_ode_dopri5

         3770321 function calls (3743370 primitive calls) in 7.942 seconds

   Ordered by: cumulative time
   List reduced from 5550 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    758/1    0.009    0.000    7.943    7.943 {built-in method builtins.exec}
        1    0.000    0.000    7.943    7.943 examples/compare_performance_ivp_burgers_eq.py:1(<module>)
        1    0.000    0.000    7.141    7.141 examples/compare_performance_ivp_burgers_eq.py:96(run_one_impl)
        1    0.003    0.003    7.141    7.141 examples/compare_performance_ivp_burgers_eq.py:194(_run_once)
     2000    0.004    0.000    6.742    0.003 oif/interfaces/python/oif/interfaces/ivp.py:38(integrate)
     2002    0.071    0.000    6.739    0.003 oif/lang_python/oif/core.py:138(call)
     2000    0.006    0.000    6.602    0.003 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:39(integrate)
     2000    0.004    0.000    6.595    0.003 <>/lib/python3.12/site-packages/scipy/integrate/_ode.py:397(integrate)
     2000    0.371    0.000    6.590    0.003 <>/lib/python3.12/site-packages/scipy/integrate/_ode.py:1173(run)
    89872    0.097    0.000    6.220    0.000 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:44(_rhs_fn_wrapper)
    89872    0.036    0.000    6.123    0.000 oif_impl/python/oif/callback.py:27(__call__)
    89872    0.560    0.000    6.087    0.000 {built-in method callback.call_c_fn_from_python}
    89872    0.699    0.000    5.527    0.000 oif/lang_python/oif/core.py:105(wrapper)
    89872    2.864    0.000    3.673    0.000 examples/compare_performance_ivp_burgers_eq.py:73(compute_rhs)
   179744    0.282    0.000    1.086    0.000 <>/lib/python3.12/site-packages/numpy/ctypeslib.py:506(as_array)
```

we can see now that the `__call__` takes about 0.59 seconds (see
`callback.callc_fn_from_python`).

This gives about 3.5 times of performance boost from the previous version.

# 2024-02-09 Conversion of OIF arrays to NumPy arrays via C extension

I have implemented a C extension to convert `OIFArrayF64 *` to NumPy arrays
not via `np.ctypeslib` but directly for efficiency.

These are the results of profiling:
```
Fri Feb  9 17:30:26 2024    profiler-results-oif-scipy_ode_dopri5

         3051649 function calls (3024692 primitive calls) in 6.099 seconds

   Ordered by: cumulative time
   List reduced from 5551 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    758/1    0.009    0.000    6.100    6.100 {built-in method builtins.exec}
        1    0.000    0.000    6.100    6.100 examples/compare_performance_ivp_burgers_eq.py:1(<module>)
        1    0.000    0.000    5.273    5.273 examples/compare_performance_ivp_burgers_eq.py:96(run_one_impl)
        1    0.003    0.003    5.273    5.273 examples/compare_performance_ivp_burgers_eq.py:194(_run_once)
     2000    0.005    0.000    4.856    0.002 oif/interfaces/python/oif/interfaces/ivp.py:38(integrate)
     2002    0.071    0.000    4.853    0.002 oif/lang_python/oif/core.py:146(call)
     2000    0.006    0.000    4.712    0.002 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:39(integrate)
     2000    0.004    0.000    4.706    0.002 <..>/lib/python3.12/site-packages/scipy/integrate/_ode.py:397(integrate)
     2000    0.351    0.000    4.701    0.002 <..>/lib/python3.12/site-packages/scipy/integrate/_ode.py:1173(run)
    89872    0.088    0.000    4.350    0.000 oif_impl/impl/ivp/scipy_ode_dopri5/dopri5.py:44(_rhs_fn_wrapper)
    89872    0.036    0.000    4.263    0.000 oif_impl/python/oif/callback.py:27(__call__)
    89872    0.413    0.000    4.227    0.000 {built-in method callback.call_c_fn_from_python}
    89872    0.410    0.000    3.813    0.000 oif/lang_python/oif/core.py:110(wrapper)
    89872    2.597    0.000    3.259    0.000 examples/compare_performance_ivp_burgers_eq.py:73(compute_rhs)
```

We can see from these results that the run time decreased in comparison
to the previous results (in the previous section), as in the `wrapper`
function conversion of OIF arrays to NumPy arrays is done in a more efficient
manner.

Native profiling results are
```
Fri Feb  9 17:49:50 2024    profiler-results-native-scipy_ode_dopri5

         2051233 function calls (2024295 primitive calls) in 4.467 seconds

   Ordered by: cumulative time
   List reduced from 5518 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    756/1    0.010    0.000    4.469    4.469 {built-in method builtins.exec}
        1    0.000    0.000    4.469    4.469 examples/compare_performance_ivp_burgers_eq.py:1(<module>)
        1    0.001    0.001    3.587    3.587 examples/compare_performance_ivp_burgers_eq.py:97(run_one_impl)
        1    0.004    0.004    3.586    3.586 examples/compare_performance_ivp_burgers_eq.py:195(_run_once)
     2000    0.004    0.000    3.185    0.002 <..>/lib/python3.12/site-packages/scipy/integrate/_ode.py:397(integrate)
     2000    0.305    0.000    3.181    0.002 <..>/lib/python3.12/site-packages/scipy/integrate/_ode.py:1173(run)
    89871    0.061    0.000    2.876    0.000 examples/compare_performance_ivp_burgers_eq.py:92(compute_rhs_native)
    89871    2.268    0.000    2.815    0.000 examples/compare_performance_ivp_burgers_eq.py:74(compute_rhs)
   849/11    0.005    0.000    1.026    0.093 <frozen importlib._bootstrap>:1349(_find_and_load)
   844/11    0.004    0.000    1.025    0.093 <frozen importlib._bootstrap>:1304(_find_and_load_unlocked)
   804/14    0.003    0.000    1.022    0.073 <frozen importlib._bootstrap>:911(_load_unlocked)
   683/12    0.002    0.000    1.022    0.085 <frozen importlib._bootstrap_external>:988(exec_module)
  2013/24    0.002    0.000    1.019    0.042 <frozen importlib._bootstrap>:480(_call_with_frames_removed)
   668/79    0.001    0.000    0.903    0.011 {built-in method builtins.__import__}
 1017/106    0.002    0.000    0.895    0.008 <frozen importlib._bootstrap>:1390(_handle_fromlist)
    89953    0.093    0.000    0.533    0.000 <..>/lib/python3.12/site-packages/numpy/core/fromnumeric.py:2692(max)
        1    0.000    0.000    0.482    0.482 <..>/lib/python3.12/site-packages/matplotlib/pyplot.py:1(<module>)
    90055    0.143    0.000    0.441    0.000 <..>/lib/python3.12/site-packages/numpy/core/fromnumeric.py:71(_wrapreduction)
1953/1833    0.027    0.000    0.433    0.000 {built-in method builtins.__build_class__}
      105    0.003    0.000    0.301    0.003 <..>/lib/python3.12/site-packages/matplotlib/artist.py:159(_update_set_signature_and_docstring)
```

Comparing the run time for `ode.integrate` with native performance (results are
above), we see now the following profiling results:

OIF or native | Run time, seconds |
oif | 4.706 |
native | 3.185 |

which gives 47% performance penalty.

# 2024-02-08 Performance study of conversion to C types from Python types

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
        # Omitted as irrelevant here.

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

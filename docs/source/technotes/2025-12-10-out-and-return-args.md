# Extended Dispatch/Bridge API

So far, we have an API like this:
```
int call_interface_impl(implh, func_name, in_args, out_args);
```
where `in_args` and `out_args` are arrays of input and output arguments.
Output arguments are provided by the caller and the callee writes into them.

Now we want an extended API:
```
int call_interface_impl(implh, func_name, in_args, out_args, return_args)
```
where `return_args` is a semi-filled array: it contains the data types
and the number of arguments, but it is Bridge that allocates the memory
and writes into this arguments passing the ownership back to the caller
(a Converter to be precise) that converts it the return data again
from the C intermediate representation to the native language of the user.

# Data passing and function calls

Copying data, especially, large arrays, impedes performance,
as modern computer architectures are bounded by memory operations.
Hence, in _Open Interfaces_ we avoid copying data and pass
all data as pointers, which makes all conversion operations fast and cheap.
Conversion of integer and floating-point numbers is cheap by itself,
but even for arrays, it is a matter of creating a thin wrapper
around an~actual data pointer.

To invoke functions between different languages,
the [Foreign Function Interface (FFI) library libffi](https://sourceware.org/libffi/)
library is used, either indirectly, e.g.,
using `ctypes` in Python,
or explicitly for calling C functions dynamically.

Note that we use C convention of functions returning an integer to indicate
an error occurred during a function call.
When the resultant integer is zero, the function invocation is successful,
and not otherwise.
For languages that support exceptions as the error-handling mechanism,
an exception is raised on the user side,
so that the user does not have to check every function call for errors.

# Data types

As stated before, we use C data types for intermediate representation, as C
is the _lingua franca_ of programming languages,
and they all have facilities to communicate with C
and making function calls to C.
Also, popular languages such as Python and Julia
have a C API that provides means of conversion of the data from the intermediate
representation to the native data types of these languages.

Particularly, it is easy to convert Python and Julia integer data types to
C `int` data type, provided that the integers are representable
in 32 bits.
Also, conversion of binary double-precision floating-point numbers
is straightforward between C and other languages due to the widespread use
of the [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754)
for floating-point arithmetic.

Data marshalling of arrays of double-precision floating-point numbers
is made possible by using an auxiliary data structure `OIFArrayF64` that,
similarly to NumPy arrays or Julia arrays,
represents $n$-dimensional array for given
$n \in \mathbb N$ and packs data together with the number of dimensions $n$
and the array shape, that is, the size of the array along each dimension.
This data structure enables a~uniform function signature
among supported languages (currently C, Python, and Julia)
as then the arrays are given as a single
function argument in all these languages (in contrast with traditional use
of C arrays, where data and dimensions are provided as separate arguments).
Correspondingly, we use NumPy C API and Julia C API to convert to
`OIFArrayF64` and back when needed.

We also support read-only strings that can be used to pass information such
as, e.g., a name of an integrator.

As it is common in scientific computing to pass callback functions to numerical
solvers, Open Interfaces support passing functions between different languages.
This is achieved in the following manner.
Additional data structure `OIFCallback` is used that encodes information
about the original language of the callback function, the function itself
in this language, and the C-compatible version of this function.
Consequently, on the language-specific dispatch level, if the user-facing
and implementation languages are the same, the original callback function
is used to avoid performance penalties, while the C-compatible callback
is wrapped in the programming language of the implementation.

Additionally to the callback, passing a generic memory pointer is supported
which is required, for example, to pass context to the callback functions.
Although in languages like Python one can simply use closures
to pass the context, in languages like C it is the only way to achieve this.

Finally, simple dictionaries of key-value pairs, where keys are strings,
and values are either integer or floats, are supported to pass generic
options that are implementation-specific.

For each supported data type, the data are passed between software components
along with integer identifiers allowing to restore the type on the receiver
end. We use the following symbolic constants
further in the text to refer to the actual data types:
 - `OIF_TYPE_I32` (or `OIF_TYPE_INT` as an alias): 32-bit integers,
 - `OIF_TYPE_F64`: 64-bit binary floating-point numbers,
 - `OIF_TYPE_ARRAY_F64`: arrays of 64-bit binary floating-point numbers,
 - `OIF_TYPE_STRING`: strings with one-byte characters
 - `OIF_TYPE_CALLBACK`: callback functions,
 - `OIF_USER_DATA`: user-data objects of volatile type,
 - `OIF_CONFIG_DICT`: dictionary of key-value options pairs.

It is assumed that each symbolic constant is replaced with the actual data type
when used in a particular language: for example, `OIF_TYPE_ARRAY_F64`
resolves to the provided data structure `OIFArrayF64` in C
and to NumPy arrays with `dtype=numpy.float64` in Python.

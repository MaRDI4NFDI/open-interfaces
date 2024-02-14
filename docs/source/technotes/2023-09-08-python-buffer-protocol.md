# 2024-09-08 Python buffer protocol

Notes on the Python buffer documentation page:
<https://docs.python.org/3/c-api/buffer.html>.

## Introduction

Many Python objects operate with large consecutive memory blocks,
or **buffers**.
Often it is necessary to pass these objects between different libraries
without copying the data.
This is what the buffer protocol is for.
It allows to expose the structure of the buffer and access the pointer
to the buffer data.

The protocol allows to expose data in read-only form or with write access.

For example, buffer interface allows to pass large arrays from Python
to a C extension module, or write objects to files.

## Buffer structure

Buffer is not `PyObject *` but data structure `Py_buffer` with many members
such as:

- `void *b`: A pointer to any location within the memory block owned by the
  buffer
- `Py_ssize_t len`: number of elements in the buffer in bytes
- `int readonly` flag indicating whether it is possible to modify the buffer
- `Py_ssize_t itemsize` size of one element of the buffer in bytes
- `int ndim` number of dimensions when the buffer is an $n$-dimensional array.
  If `0` then the buffer is scalar.
- `Py_ssize_t *shape` size of each dimension in an $n$-dimensional array
- `Py_ssize_t *strides` an array where each element indicates the number of
  bytes that should be skipped when moving to another element in each
  direction.

When working with buffers, it is possible to ask whether the buffer is
contiguous or not.
Also, it is possible to know about the memory layout (C style or Fortran
style).

## Complex arrays

### NumPy arrays

NumPy arrays can be described completely with four properties:
`itemsize`, `ndim`, `shape`, and `strides`.

If `strides` is `NULL`, then it is just contiguous array with C memory layout.

If strides is not `NULL` (to remind, strides are specified in bytes),
then, for example, item `[i, j]` in a two-dimensional array with elements
of type `double` can be accessed via:

```c
ptr_i_j = ((char *) buf) + i * strides[0] + j * strides[1]
item = *((double *) ptr_i_j)
```

## Buffer-related functions

Buffer-related functions start with prefix `PyBuffer`.

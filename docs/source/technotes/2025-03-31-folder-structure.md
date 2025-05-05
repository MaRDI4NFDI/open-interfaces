# 2025-03-31 Folder Structure

We need a solid folder structure so that different components are installed
somewhere where they can be found.


## Python

For Python package, we want to install everything in `site-packages` directory.

There are two principal components:

- `oif` - user-facing code
- `oif_impl` - implementation code, that, probably, should not be
  exposed to the user directly (maybe, `_oif_impl` then?)

One of possible structures is:
```
site-packages/
  oif/
    __init__.py
    core.py              # Converter
    _impl/
      __init__.py
      ivp.py
      ...

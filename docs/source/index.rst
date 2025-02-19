.. open-interfaces documentation master file, created by
   sphinx-quickstart on Wed Aug 30 13:04:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MaRDI Open Interfaces
=====================

*MaDRI Open Interfaces* is a software package that aims to improve
interoperability and reusability in Scientific Computing.
It does so

* by providing a set of interfaces that hide the implementation
  details of numerical packages, allowing users to switch between different
  implementations without slow and costly code modifications,
* by doing data marshalling between different
  programming languages automatically, so that users of one language
  could use implementations written in another language.

For example, using *Open Interfaces*,
computational scientists that prefer Python,
can use implementations written in C or Julia without writing explicit bindings
and easily switch between different implementations of numerical algorithms
for the same problem type.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :numbered:

   getting-started/index.rst
   how-to/index.rst
   explanation/index.rst
   api/index
   technotes/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

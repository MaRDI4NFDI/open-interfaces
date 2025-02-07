linear_solver
=============

.. py:module:: linear_solver

.. autoapi-nested-parse::

   This module defines the interface for solving linear systems of equations.

   Problems to be solved are of the form:

       .. math::
           A x = b,

   where :math:`A` is a square matrix and :math:`b` is a vector.





Module Contents
---------------

.. py:class:: LinearSolver(impl: str)

   Interface for solving linear systems of equations.

   This class serves as a gateway to the implementations of the
   linear algebraic solvers.

   :param impl: Name of the desired implementation.
   :type impl: str


   .. py:method:: solve(A: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

      Solve the linear system of equations :math:`A x = b`.

      :param A: Coefficient matrix.
      :type A: np.ndarray of shape (n, n)
      :param b: Right-hand side vector.
      :type b: np.ndarray of shape (n,)

      :returns: Result of the linear system solution after the invocation
                of the `solve` method.
      :rtype: np.ndarray




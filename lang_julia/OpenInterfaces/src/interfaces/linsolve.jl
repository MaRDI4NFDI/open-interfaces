"""This module defines the interface for solving linear systems of equations.

Problems to be solved are of the form:

    .. math::
        A x = b,

where :math:`A` is a square matrix and :math:`b` is a vector.
"""
module Linsolve

export solve

using OpenInterfaces: ImplHandle, call_impl

function solve(implh:: ImplHandle, A:: Matrix{Float64}, b:: Vector{Float64}):: Vector{Float64}
    """Solve the linear system of equations :math:`A x = b`.

    Parameters
    ----------
    A : np.ndarray of shape (n, n)
        Coefficient matrix.
    b : np.ndarray of shape (n,)
        Right-hand side vector.

    Returns
    -------
    np.ndarray
        Result of the linear system solution after the invocation
        of the `solve` method.

    """
    m, n = size(A)
    result = Vector{Float64}(undef, n)

    call_impl(implh, "solve_lin", (A, b), (result,))
    return result
end

end

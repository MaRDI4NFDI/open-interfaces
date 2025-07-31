"""This module defines the interface for solving linear systems of equations.

Problems to be solved are of the form:

    .. math::
        A x = b,

where :math:`A` is a square matrix and :math:`b` is a vector.
"""
module Linsolve

export solve

using OpenInterfaces: ImplHandle, call_impl


"""
    solve(implh, A, b) -> AbstractVector{Float64}


Solve the linear system of equations ``A x = b`` and return ``x``.

# Arguments
- `A::AbstractMatrix{Float64}`: Coefficient square matrix
- `b::AbstractVector{Float64}`: Right-hand side vector

"""
function solve(
    implh::ImplHandle,
    A::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
)::AbstractVector{Float64}
    m, n = size(A)
    result = Vector{Float64}(undef, n)

    call_impl(implh, "solve_lin", (A, b), (result,))
    return result
end

end

"""This module defines the interface for solving a quadratic equation.

The quadratic equation is of the form:

.. math::
    a x^2 + b x + c = 0,

where :math:`a`, :math:`b`, and :math:`c` are the coefficients of the equation.

Of course, this is not very useful in scientific context to invoke
such a solver.

It was developed as a prototype to ensure that the envisioned architecture
of Open Interfaces is feasible.
It is used as a simple text case as well.

"""
module QEQ

using OpenInterfaces: ImplHandle, call_impl

export solve_qeq

function solve_qeq(implh::ImplHandle, a::Float64, b::Float64, c::Float64)::Vector{Float64}
    result = [11.0, 22.0]
    call_impl(implh, "solve_qeq", (a, b, c), (result,))
    return result
end

end

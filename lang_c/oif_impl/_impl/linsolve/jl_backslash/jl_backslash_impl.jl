module JlBackslashImpl

export solve_lin

mutable struct Self end

function solve_lin(
    self::Self,
    A::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
    result::AbstractVector{Float64},
)::Int32
    result[:] = A \ b
    return 0
end

end

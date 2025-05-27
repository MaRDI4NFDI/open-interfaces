module JlBackslashImpl

export solve_lin

mutable struct Self
end

function solve_lin(self::Self, A::Matrix{Float64}, b::Vector{Float64}, result::Vector{Float64})::Int32
    result[:] = A \ b
    return 0
end

end

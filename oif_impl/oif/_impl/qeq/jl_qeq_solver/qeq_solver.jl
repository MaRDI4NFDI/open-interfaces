module QeqSolver
export solve_qeq

# This solver does not actually require any context, so Self is empty.
mutable struct Self end

function solve_qeq(self::Self, a::Float64, b::Float64, c::Float64, roots::Vector{Float64})
    println("Solving quadratic equation with a=$a, b=$b, c=$c")
    if a == 0
        roots[1] = -c / b
        roots[2] = -c / b
    else
        D = b^2 - 4 * a * c
        if b > 0
            roots[1] = (-b - sqrt(D)) / (2 * a)
            roots[2] = c / (a * roots[1])
        else
            roots[1] = (-b + sqrt(D)) / (2 * a)
            roots[2] = c / (a * roots[1])
        end
    end

    return 0
end
end

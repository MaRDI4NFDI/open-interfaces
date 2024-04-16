module QeqSolver
export solve!

function solve_qeq!(a, b, c, roots)
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

function solve!(a, b, c, roots_ptr::Ptr{Float64})::Cint
    roots = unsafe_wrap(Array, roots_ptr, 2)
    return solve!(a, b, c, roots)
end
end

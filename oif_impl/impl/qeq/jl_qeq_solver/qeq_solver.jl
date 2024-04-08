module QeqSolver
export solve

function solve!(a, b, c, roots::Ref{Array{Float64}})
    println("roots[1] = "),
    println(roots[1])
    if a == 0
        roots[1] = -c / b
        roots[2] = -c / b
    else
        D = b^2 - 4 * a * c
        println("D = ", D)
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

using LinearAlgebra

using OpenInterfaces
using OpenInterfaces.Interfaces.Linsolve


function parse_args(args)
    supported_impls = ["c_lapack", "jl_backslash", "numpy"]

    if length(args) > 0
        impl = args[1]

        if ! (impl in supported_impls)
            error("""
                  Given implementation $impl is not in the list\
                  of supported implementation: $supported_impls
            """)
            println("Hello")
        end
    else
        impl = "jl_backslash"
    end

    return impl
end


function main(args)
    impl = parse_args(args)
    println("Calling from Julia an open interface for linear algebraic systems")
    println("Implementation: $impl")
    A = [ 1.0 1.0; -3.0 1.0 ]
    b = [6.0, 2.0]
    implh = load_impl("linsolve", impl, 1, 0)
    x = Linsolve.solve(implh, A, b)

    println("Solving system of linear equations Ax = b:")
    println("A = $A")
    println("b = $b")
    println("x = $x")
    println("L2 error = $(norm(A * x - b, 2))")
end


main(ARGS)

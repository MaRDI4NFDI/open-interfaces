using Printf

using OpenInterfaces.Interfaces.IVP


function parse_args(args)
    supported_impls = ["scipy_ode", "sundials_cvode", "dopri5c", "jl_diffeq"]

    if length(args) > 0
        impl = args[1]

        if ! (impl in supported_impls)
            error("""
                  Given implementation $impl is not in the list\
                  of the supported implementations: $supported_impls
                  """)
        end
    else
        impl = "jl_diffeq"
    end

    return impl
end


function rhs(t, y, ydot, user_data)
    ydot[:] = -y
    return 0
end


function main(args)
    impl = parse_args(args)
    println("Calling from Python an open interface for initial-value problems")
    println("Implementation: $impl")

    solver = IVP.Self(impl)
    t0 = 0.0
    y0 = [1.0]
    IVP.set_initial_value(solver, y0, t0)
    IVP.set_rhs_fn(solver, rhs)

    times = range(t0, t0 + 1, length=11)

    @printf "%5s   %8s   %8s\n" "Time" "Numeric" "Exact"
    @printf "---------------------------\n"
    for t in times[2:end]
        IVP.integrate(solver, t)
        exact = exp(-t)
        @printf "%.3f   %.6f   %.6f\n" t solver.y[1] exact
    end
end


main(ARGS)

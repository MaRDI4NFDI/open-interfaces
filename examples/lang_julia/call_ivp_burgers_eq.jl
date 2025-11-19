using OpenInterfaces.Interfaces.IVP


struct Args
    impl::String
    savefig::Bool
    no_plot::Bool
end


function parse_args(args)
    supported_impls = ["scipy_ode", "sundials_cvode", "dopri5c", "jl_diffeq"]

    impl = "jl_diffeq"
    savefig = false
    no_plot = false

    if length(args) > 0
        if !startswith(args[1], "-")
            impl = args[1]

            if ! (impl in supported_impls)
                error("""
                      Given implementation $impl is not in the list\
                      of the supported implementations: $supported_impls\
                      """)
            end
        end
    end

    for arg in args
        # Save figure instead of showing it
        if arg in ["--savefig", "-s"]
            savefig = true
        end

        # Set to avoid plotting, for example, in tests
        if arg in ["--no-plot"]
            no_plot = true
        end
    end

    return Args(impl, savefig, no_plot)
end


function compute_rhs(__, u, udot, user_data)::Int32
    dx = user_data  # Directly access the first element of the tuple.
    dx_inv = inv(dx)
    N = length(udot)

    c = maximum(abs, u)  # abs, u applies abs without creating a temp array.

    f_cur = 0.5 * u[1]^2
    f_hat_lb = 0.5 * (f_cur + 0.5 * u[N]^2 - 0.5 * c * (u[1] - u[N])
    f_hat_prev = f_hat_lb
    @inbounds for i = 1:N-1
        f_next = 0.5 * u[i+1]^2
        f_hat_cur = 0.5 * ((f_cur+f_next) - c * (u[i+1]-u[i]))
        udot[i] = dx_inv * (f_hat_prev - f_hat_cur)
        f_hat_prev, f_cur = f_hat_cur, f_next
    end
    udot[N] = dx_inv * (f_hat_prev - f_hat_lb)
end


function main(args)
    args = parse_args(args)
    println("Solving Burgers' equation with IVP interface using Julia bindings")
    println("Implementation: $(args.impl)")

    # Define grid and initial condition.
    N = 3
    x = range(0, 2, length=N)
    dx = (2 - 0) / N
    u0 = 0.5 .- 0.25 * sin.(pi * x)

    t0 = 0.0
    tfinal = 2.0

    s = IVP.Self(args.impl)
    IVP.set_initial_value(s, u0, t0)
    IVP.set_user_data(s, dx)
    IVP.set_rhs_fn(s, compute_rhs)

    times = range(t0, tfinal, length=11)

#     soln = [problem.u0]
#     for t in times[1:]:
#         s.integrate(t)
#         soln.append(s.y)

#     plt.plot(problem.x, soln[0], "--", label="Initial condition")
#     plt.plot(problem.x, soln[-1], "-", label="Final solution")
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"Solution of Burgers' equation")
#     plt.legend(loc="best")
#     plt.tight_layout(pad=0.1)

#     if args.savefig:
#         plt.savefig(os.path.join("assets", f"ivp_py_burgers_eq_{impl}.pdf"))
#     else:
#         if not args.no_plot:
#             plt.show()
end


main(ARGS)

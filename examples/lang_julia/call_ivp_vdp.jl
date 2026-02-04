using OpenInterfaces.Interfaces.IVP
using Plots


struct Args
    impl::String
    savefig::Bool
    no_plot::Bool
end


function parse_args(args)
    supported_impls = [
        "scipy_ode-dopri5",
        "scipy_ode-dopri5-100k",
        "scipy_ode-vode",
        "scipy_ode-vode-40k",
        "sundials_cvode-default",
        "jl_diffeq-rosenbrock23",
    ]

    impl = "jl_diffeq-rosenbrock23"
    savefig = false
    no_plot = false

    if length(args) > 0
        if !startswith(args[1], "-")
            impl = args[1]

            if !(impl in supported_impls)
                error("""
                      Given implementation '$impl' is not in the list \
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
    mu = user_data  # Directly access the first element of the tuple.
    udot[1] = u[2]
    udot[2] = mu * (1 - u[1]^2) * u[2] - u[1]
    return 0
end


function main(args)
    args = parse_args(args)
    chunks = split(args.impl, "-"; limit = 2)
    impl = String(chunks[1])
    integrator = String(chunks[2])
    println("Solving van der Pol oscillator with IVP interface using Julia bindings")
    println("Implementation: $(impl)")
    println("Integrator: $(integrator)")

    # Define grid and initial condition.
    N = 1001
    mu = 1000
    u0 = [2.0, 0.0]

    t0 = 0.0
    tfinal = 3000

    s = IVP.Self(impl)
    IVP.set_initial_value(s, u0, t0)
    IVP.set_user_data(s, mu)
    IVP.set_rhs_fn(s, compute_rhs)
    IVP.set_tolerances(s, 1e-8, 1e-12)

    if impl == "sundials_cvode"
        IVP.set_integrator(s, "bdf", Dict([("max_num_steps", 30_000)]))
    elseif impl == "scipy_ode" && integrator == "dopri5"
        IVP.set_integrator(s, "dopri5", Dict())  # It is already the default integrator
    elseif impl == "scipy_ode" && integrator == "dopri5-100k"
        IVP.set_integrator(s, "dopri5", Dict([("nsteps", 100_000)]))
    elseif impl == "scipy_ode" && integrator == "vode"
        IVP.set_integrator(s, "vode", Dict([("method", "bdf")]))
    elseif impl == "scipy_ode" && integrator == "vode-40k"
        IVP.set_integrator(s, "vode", Dict([("method", "bdf"), ("nsteps", 40_000)]))
    elseif impl == "jl_diffeq" && lowercase(integrator) == lowercase("Rosenbrock23")
        IVP.set_integrator(s, "Rosenbrock23", Dict([("autodiff", false),]))
    else
        throw(ArgumentError(`Cannot set integrator for implementation '$impl'`))
    end

    times = range(t0, tfinal, length = 501)
    dt = (tfinal - t0) / length(times)

    soln = [u0[1]]
    for t in times[2:end]
        if round(t / dt) % 10 == 0
            print(".")
        end
        IVP.integrate(s, t)
        push!(soln, s.y[1])
    end
    println()

    p = plot(times, soln, linestyle = :solid, label = "\$u_1\$", linewidth = 2)
    xlabel!(p, "Time")
    ylabel!(p, "Solution")
    plot!(p, legend = :best)

    if !args.no_plot
        if args.savefig
            savefig(p, joinpath("assets", "ivp_jl_vdp_eq_$(args.impl).pdf"))
        else
            println("Displaying figure")
            display(p)
            if !Base.isinteractive()
                print("Press Enter to quit: ")
                readline()
            end
        end
    end
end


main(ARGS)

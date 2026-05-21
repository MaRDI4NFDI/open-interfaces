using Printf

using OpenInterfaces.Interfaces.Optim


function parse_args(args)
    supported_impls = ["optim_jl", "ipopt_jl", "scipy_optimize"]

    impl = "optim_jl"
    method = "NelderMead"
    linesearch = "StrongWolfe"

    if length(args) > 0
        impl = args[1]

        if ! (impl in supported_impls)
            error("""
                  Given implementation $impl is not in the list\
                  of the supported implementations: $supported_impls
                  """)
        end
    end

    if length(args) > 1
        method = args[2]
    end

    if length(args) > 2
        linesearch = args[3]
    else
        linesearch
    end

    return impl, method, linesearch
end


function rosenbrock_objective_fn(x, a)
    """The Rosenbrock function with additional arguments"""
    return sum(a * (x[2:end] - x[1:(end-1)] .^ 2.0) .^ 2.0 + (1 .- x[1:(end-1)]) .^ 2.0)
end

function rosenbrock_grad_fn(x, grad_f, a)
    """Gradient of the Rosenbrock function"""

    # `xi` is x[i], `xip1` is x[i+1]
    xi = @view x[1:(end-1)]
    xip1 = @view x[2:end]

    grad_f[1:(end-1)] .= -4.0 .* a .* xi .* (xip1 .- xi .^ 2.0) .- 2.0 .* (1.0 .- xi)
    grad_f[end] = 0.0
    grad_f[2:end] .+= 2.0 .* a .* (xip1 .- xi .^ 2.0)
    return 0
end


function main(args)
    impl, method, linesearch = parse_args(args)
    println("Calling from Julia an open interface for initial-value problems (IVP)")
    println("Implementation: $impl")
    println("Method: $method")
    println("Line search (only for optim_jl:BFGS): $linesearch")

    x0 = [3.14, 2.72, 6.18, 9.81, 8.31]
    user_data = 10

    s = Optim.Self(impl)
    Optim.set_initial_guess(s, x0)
    Optim.set_user_data(s, user_data)
    if impl == "scipy_optimize"
        if method == "NelderMead"
            Optim.set_method(s, "nelder-mead", Dict("fatol" => 1e-11))
        elseif method == "BFGS"
            Optim.set_method(s, "BFGS", Dict("gtol" => 1e-8))
        else
            error("Unsupported method '$(method)'")
        end
    elseif impl == "optim_jl"
        if method == "NelderMead"
            Optim.set_method(s, "NelderMead", Dict("g_abstol" => 1e-11))
        elseif method == "BFGS"
            Optim.set_method(
                s,
                "BFGS",
                Dict("g_abstol" => 1e-8, "linesearch" => linesearch),
            )
        else
            error("Unsupported method '$(method)'")
        end
    else
        error("Unknown implementation")
    end
    Optim.set_objective_fn(s, rosenbrock_objective_fn)
    Optim.set_grad_fn(s, rosenbrock_grad_fn)

    status, message = Optim.minimize(s)
    x = s.x

    println("Message: ", message)
    @assert status == 0
    println("x = ", x)
    if all(abs.(x .- 1.0) .< 1e-5)  # The solution is [1, 1, ..., 1].
        println("\033[1;32mSUCCESS\033[0m Found solution is close to the exact one")
    else
        println("\033[1;31mFAIL\033[0m Found solution is NOT close to the exact one")
    end
end


main(ARGS)

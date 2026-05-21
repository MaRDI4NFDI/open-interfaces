# This module is an adapter to the Optim.jl package.
module OptimJl

export Self, set_initial_guess


using LineSearches
using Optim

GENERAL_OPTIONS_NAMES = [
    :x_abstol,
    :x_reltol,
    :x_tol,  # Deprecated, keep to be user-friendly
    :f_abstol,
    :f_reltol,
    :f_tol,  # Deprecated, keep to be user-friendly
    :g_abstol,
    :g_tol,  # Deprecated, keep to be user-friendly
    :f_calls_limit,
    :g_calls_limit,
    :h_calls_limit,
    :allow_f_increases,
    :successive_f_tol,
    :iterations,
    :time_limit,
    :callback,
]


mutable struct Self
    x0::Vector{Float64}
    objective_fn::Union{Function,Nothing}
    grad_fn::Union{Function,Nothing}
    user_data::Any
    method_name::String
    method_options::Dict
    method::Any
    general_options::Dict
    function Self()
        return new([], nothing, nothing, nothing, "BFGS", Dict(), BFGS(), Dict())
    end
end

function set_initial_guess(self::Self, x0)
    self.x0 = x0
    return nothing
end

function set_user_data(self::Self, user_data)
    self.user_data = user_data
end

function set_objective_fn(self::Self, objective_fn)
    self.objective_fn = objective_fn
    self.objective_fn(self.x0, self.user_data)
end

function set_grad_fn(self::Self, grad_fn)
    self.grad_fn = grad_fn
    self.grad_fn(self.x0, zeros(length(self.x0)), self.user_data)
end

function set_method(self, method_name, method_params)
    println(
        "[optim::optim_jl] To check available configuration options, " *
        "see https://julianlsolvers.github.io/Optim.jl/stable/user/config/",
    )
    general_options = Dict()
    method_options = Dict()

    if haskey(method_params, :linesearch)
        method_options[:linesearch] =
            getfield(LineSearches, Symbol(method_params[:linesearch]))()
        delete!(method_params, :linesearch)
    end

    for (k, v) in method_params
        if k in GENERAL_OPTIONS_NAMES
            general_options[k] = v
        else
            method_options[k] = v
        end
    end

    method_symbol = Symbol(method_name)
    if (!isdefined(Optim, method_symbol))
        error("Unknown or unsupported method '$method_name'")
    end
    self.method_name = method_name
    self.method_options = merge(self.method_options, method_options)
    self.method = getfield(Optim, method_symbol)(; method_options...)
    self.general_options = general_options
    println("method_options = ", method_options)
    println("general_options = ", general_options)
end

function minimize(self::Self, out_x)::Tuple{Int,String}
    if !isdefined(self, :user_data)
        self.user_data = nothing
    end
    N = length(self.x0)

    wrapper(x) = self.objective_fn(x, self.user_data)

    result::Optim.MultivariateOptimizationResults = begin
        if isnothing(self.grad_fn)
            optimize(wrapper, self.x0, self.method, Optim.Options(; self.general_options...))
        else
            g_wrapper(grad_values, x) = self.grad_fn(x, grad_values, self.user_data)
            optimize(
                wrapper,
                g_wrapper,
                self.x0,
                self.method,
                Optim.Options(; self.general_options...),
            )
        end
    end
    out_x[:] = copy(Optim.minimizer(result))

    println("res = ", result)

    if Optim.converged(result) == true
        status = 0
    else
        status = -1
    end

    if Optim.converged(result) == true
        message = "The algorithm has converged"
    else
        message = "The algorithm has not converged"
    end

    return (status, message)
end


end  # module IpoptJl

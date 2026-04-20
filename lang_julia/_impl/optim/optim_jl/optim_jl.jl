# This module is an adapter to the Optim.jl package.
module OptimJl

export Self, set_initial_guess


using Optim


mutable struct Self
    x0::Vector{Float64}
    objective_fn::Union{Function,Nothing}
    user_data::Any
    method_name::String
    method_params::Dict
    method::Any
    function Self()
        return new([], nothing, nothing, "BFGS", Dict(), BFGS())
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
end

function set_method(self, method_name, method_params)
    general_options = Dict()
    method_options = Dict()

    for (k, v) in method_params
        if k in general_options_names
            general_options[k] = v
        else
            method_options[k] = v
        end
    end

    method_symbol = Symbol(method_name)
    if (!isdefined(Optim, method_symbol))
        error("Unknown or unsupported method '$method_name'. Supported methods are: $supported_methods")
    end
    self.method_name = method_name
    self.method_params = merge(self.method_params, method_params)
    self.method = getfield(Optim, method_symbol)(; method_options)
end

function minimize(self::Self, out_x)::Tuple{Int,String}
    if !isdefined(self, :user_data)
        self.user_data = nothing
    end
    N = length(self.x0)

    wrapper(x) = self.objective_fn(x, self.user_data)

    result::Optim.MultivariateOptimizationResults = optimize(wrapper, self.x0, self.method)
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
        message =
            "The algorithm has not converged"
    end

    return (status, message)
end


end  # module IpoptJl

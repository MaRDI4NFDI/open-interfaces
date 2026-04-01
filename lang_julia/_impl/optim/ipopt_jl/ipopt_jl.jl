module IpoptJl

export Self, set_initial_guess


# JuMP = Julia Mathematical Programming package
using JuMP
using Ipopt


mutable struct Self
    x0::Vector{Float64}
    objective_fn::Union{Function,Nothing}
    user_data::Any
    method_params::Dict
    function Self()
        return new([], nothing, nothing, Dict())
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
    println("[ipopt_jl] `method_name` is not supported by this implementation")

    # To check that we have valid parameters,
    # we use a simple model and optimize a convex problem on it.
    # This is needed, as the actual checks on the parameters
    # is done in `optimize!` in `Ipopt.jl`.
    guinea_pig = Model(Ipopt.Optimizer)
    set_silent(guinea_pig)
    @variable(guinea_pig, x_[1:2])
    @objective(guinea_pig, Min, x_[1]^2 + x_[2]^2)
    for (key, value) in method_params
        attr = String(key)
        set_attribute(guinea_pig, attr, value)
    end
    optimize!(guinea_pig)

    # If we reach this point, this means that passed `method_params`
    # are allowed parameters.
    self.method_params = merge(self.method_params, method_params)
end

function minimize(self::Self, out_x)::Tuple{Int,String}
    if !isdefined(self, :user_data)
        self.user_data = nothing
    end
    N = length(self.x0)
    model = Model(Ipopt.Optimizer)
    x = @variable(model, [1:N])

    set_start_value.(x, self.x0)
    wrapper(x) = self.objective_fn(x, self.user_data)
    @objective(model, Min, wrapper(x))

    # Set options (`method_params`) that were passed earlier in `set_method`.
    for (key, value) in self.method_params
        attr = String(key)
        set_attribute(model, attr, value)
    end

    optimize!(model)
    out_x .= value.(x)

    jump_status = termination_status(model)
    if jump_status == OPTIMAL || jump_status == LOCALLY_SOLVED
        status = 0
    else
        status = -1
    end

    if jump_status == OPTIMAL
        message = "The algorithm found a globally optimal solution."
    else
        message =
            "The algorithm converged to a stationary point, " *
            "local optimal solution, " *
            "could not find directions for improvement, " *
            "or otherwise completed its search without global guarantees."
    end

    return (status, message)
end


end  # module IpoptJl

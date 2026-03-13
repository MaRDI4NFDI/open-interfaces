module JlOptim

export Self, set_initial_guess


# JuMP = Julia Mathematical Programming package
using JuMP
using Ipopt


mutable struct Self
    x0::Vector{Float64}
    objective_fn::Function
    user_data::Any
    function Self()
        return new()
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
    throw(MethodError())
end

function minimize(self::Self, out_x)::Tuple{Int, String}
    if !isdefined(self, :user_data)
        self.user_data = nothing
    end
    model = Model(Ipopt.Optimizer)
    N = length(self.x0)
    @variable(model, x[1:N])
    set_start_value.(x, self.x0)
    wrapper(x) = self.objective_fn(x, self.user_data)
    @objective(model, Min, wrapper(x))
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
        message = "The algorithm converged to a stationary point, " *
            "local optimal solution, " *
            "could not find directions for improvement, " *
            "or otherwise completed its search without global guarantees."
    end

    return (status, message)
end


end  # module JlOptim

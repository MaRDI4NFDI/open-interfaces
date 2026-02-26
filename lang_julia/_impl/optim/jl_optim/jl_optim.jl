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
    println("=== debug == x0 = ", x0)
    self.x0 = x0
    return nothing
end

function set_user_data(self::Self, user_data)
    self.user_data = user_data
end

function set_objective_fn(self::Self, objective_fn)
    println("=== debug === objective_fn = ", objective_fn)
    self.objective_fn = objective_fn
end

function set_method(self, method_name, method_params)
    throw(MethodError())
end

function minimize(self::Self, out_x)::Tuple{Int, String}
    model = Model(Ipopt.Optimizer)
    @variable(model, x, start = self.x0)
    @objective(model, Min, self.objective_fn)
    optimize!(model)

    jump_status = termination_status(model)
    if jump_status == OPTIMAL
        status = 0
    else
        status = -1
    end

    if jump_status == OPTIMAL
        message = "Optimization was successful"
    else
        message = "Optimization has failed"
    end

    return (status, message)
end


end  # module JlOptim

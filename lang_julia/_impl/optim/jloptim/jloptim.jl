module JlOptim

export Self, set_initial_guess


# JuMP = Julia Mathematical Programming package
using JuMP
using Ipopt


mutable struct Self
    x0::Vector{Float64}
    function Self()
        return new()
    end
end

function set_initial_guess(self::Self, x0)
    self.x0 = x0
end

function set_user_data(self::Self, user_data)
    self.user_data = user_data
end

function set_objective_fn(self::Self, objective_fn)
    self.objective_fn = objective_fn
end

function set_method(self, method_name, method_params)
    throw RuntimeError()
end

function minimize(self::Self, out_x)::Tuple{Int, String}
    model = Model(Ipopt.optimizer)
    @variable(model, x, start = self.x0)
    @objective(model, Min, self.objective_fn)
    optimize!(model)


    return (status, message)
end


end  # module JlOptim

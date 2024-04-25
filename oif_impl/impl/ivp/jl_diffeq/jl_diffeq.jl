module JlDiffEq
export Self, set_initial_value, set_rhs_fn, set_tolerances, integrate, set_user_data

using OrdinaryDiffEq: ODEProblem, Tsit5, init

mutable struct Self
    t0::Float64
    y0::Vector{Float64}
    rhs
    problem
    s
    user_data::Tuple
    function Self()
        return new(0.0, [])
    end
end

function set_initial_value(self::Self, y0, t0)
    println("I am setting initial value")
    self.t0 = t0
    self.y0 = y0
    println("Initial time is = ", self.t0)
    return 0
end

function set_rhs_fn(self::Self, rhs)
    self.rhs = rhs

    if !isempty(self.y0)
        if !isdefined(self, :user_data)
            println("I am here")
            self.problem = ODEProblem(
                rhs_wrapper(rhs), self.y0, (self.t0, Inf)
            )
        else
            println("I am there")
            self.problem = ODEProblem(
                rhs_wrapper(rhs), self.y0, (self.t0, Inf), self.user_data
            )
        end
        println("Do I come here?")
        println(self.problem)
        self.s = init(self.problem, Tsit5())
    else
        throw(MethodError("Method `set_initial_value` must be called before `set_rhs_fn`"))
    end
    println("End of set_rhs")
end

function set_tolerances(self::Self, rtol::Float64, atol::Float64)::Int
    println("I am setting tolerances")
    return 0
end

function integrate(self::Self, t::Float64, y::Vector{Float64})::Int
    println("I am integrating")
    step!(self.integrator, t - self.integrator.t, true)
    y[:] = self.integrator.u
    return 0
end

function set_user_data(self::Self, user_data)::Int
    println("I am setting user data")
    self.user_data = user_data
end

function rhs_wrapper(rhs)
    function wrapper(du, u, p, t)
        return rhs(t, u, du, p)
    end
    return wrapper
end

end

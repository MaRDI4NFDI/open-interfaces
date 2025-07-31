module JlDiffEq
export Self,
    set_initial_value, set_rhs_fn, set_tolerances, integrate, set_user_data, set_integrator

using OrdinaryDiffEq

mutable struct Self
    t0::Float64
    y0::Vector{Float64}
    reltol::Float64
    abstol::Float64
    integrator::Any
    rhs::Any
    solver::Any
    user_data::Any
    function Self()
        return new(0.0, [], 1e-6, 1e-12, Tsit5())
    end
end

function set_initial_value(self::Self, y0, t0)::Int
    self.t0 = t0
    self.y0 = y0
    return 0
end

function set_rhs_fn(self::Self, rhs)::Int
    self.rhs = rhs

    if !isempty(self.y0)
        _init(self)
    else
        error("Method `set_initial_value` must be called before `set_rhs_fn`")
    end
    return 0
end

function set_tolerances(self::Self, reltol::Float64, abstol::Float64)::Int
    self.reltol = reltol
    self.abstol = abstol
    if !isempty(self.y0) && isdefined(self, :rhs)
        _init(self)
    end
    return 0
end

function integrate(self::Self, t::Float64, y::Vector{Float64})::Int
    step!(self.solver, t - self.solver.t, true)
    y[:] = self.solver.u
    return 0
end

function set_user_data(self::Self, user_data)
    self.user_data = user_data
    return 0
end

function set_integrator(self::Self, integrator_name::String, params::Dict)
    integrator_symbol = Symbol(integrator_name)
    if !isdefined(OrdinaryDiffEq, integrator_symbol)
        error("[jl_diffeq] Could not find integrator '$integrator_name'")
    end
    self.integrator = getfield(OrdinaryDiffEq, integrator_symbol)(; params...)
    if !isempty(self.y0) && isdefined(self, :rhs)
        _init(self)
    end
    return 0
end

function print_stats(self::Self)
    println("Number of RHS evals =", self.solver.stats.nf)
end

function _rhs_wrapper(rhs)
    # From the documentation:
    # with `ODEProblem`s the function signature is always
    # `f(du,u,p,t)` for the in-place form
    # or `f(u,p,t)` for the out-of-place form.
    # Note that the `p` argument will always be in the function signature
    # regardless of if the problem is defined with parameters!
    function wrapper(du, u, p, t)
        return rhs(t, u, du, p)
    end
    return wrapper
end

function _init(self::Self)
    if !isdefined(self, :user_data)
        problem = ODEProblem(_rhs_wrapper(self.rhs), self.y0, (self.t0, Inf))
    else
        problem =
            ODEProblem(_rhs_wrapper(self.rhs), self.y0, (self.t0, Inf), self.user_data)
    end
    self.solver = init(
        problem,
        self.integrator;
        reltol = self.reltol,
        abstol = self.abstol,
        save_everystep = false,
    )
end
end

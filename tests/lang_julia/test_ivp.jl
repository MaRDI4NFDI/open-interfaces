using LinearAlgebra
using NonlinearSolve
using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP


struct IVPProblem
    t0::Float64
    y0::Vector{Float64}
    rhs::Function
    exact::Function
end


function ScalarExpDecayProblem()
    t0 = 0.0
    y0 = [1.0]
    function rhs(__, y, ydot, __)
        ydot[1] = -y[1]
        return 0
    end
    exact = (t) -> y0 * exp(-t)

    IVPProblem(t0, y0, rhs, exact)
end

function LinearOscillatorProblem()
    t0 = 0.0
    y0 = [1.0, 0.5]
    omega = π

    function rhs(__, y, ydot, __)
        ydot[1] = y[2]
        ydot[2] = -(omega^2) * y[1]
        return 0
    end

    function exact(t)
        [
            y0[1] * cos(omega * t) +
            + y0[2] * sin(omega * t) / omega,
            -y0[1] * omega * sin(omega * t) +
            + y0[2] * cos(omega * t)
        ]
    end

    IVPProblem(t0, y0, rhs, exact)
end


function OrbitEquationsProblem()
    p_eps = 0.9
    t0 = 0.0
    y0 = [1 - p_eps, 0.0, 0.0, sqrt((1 + p_eps) / (1 - p_eps))]

    function rhs(_, y, ydot, __)
        r = sqrt(y[1]^2 + y[2]^2)
        ydot[1] = y[3]
        ydot[2] = y[4]
        ydot[3] = -y[1] / r^3
        ydot[4] = -y[2] / r^3
        return 0
    end

    function exact(t)
        function f(u, p)
            p_eps, = p
            return u - p_eps * sin.(u) .- t
        end

        nonlin_prob = NonlinearSolve.NonlinearProblem(f, [1.0], (p_eps,))
        sol = NonlinearSolve.solve(nonlin_prob)
        @assert sol.retcode == ReturnCode.Success
        u = sol.u[1]  # Extract the scalar value from the array.

        return [
            cos(u) - p_eps,
            sqrt(1 - p_eps^2) * sin(u),
            -sin(u) / (1 - p_eps * cos(u)),
            (sqrt(1 - p_eps^2) * cos(u)) / (1 - p_eps * cos(u)),
        ]
    end

    IVPProblem(t0, y0, rhs, exact)
end


function IVPProblemWithUserData()
    t0 = 0.0
    y0 = [0.0, 1.0]

    function rhs(_, y, ydot, (a, b)::Tuple{Int, Float64})
        ydot[1] = y[1] + a
        ydot[2] = b * y[2]
        return 0
    end

    function exact(t, (a, b)::Tuple{Int, Float64})
        return [
            a * (exp(t) - 1),
            exp(b * t),
        ]
    end

    IVPProblem(t0, y0, rhs, exact)
end



IMPLEMENTATIONS = ["sundials_cvode", "jl_diffeq", "scipy_ode"]

INTEGRATORS = Dict(
    "sundials_cvode" => ["adams", "bdf"],
    "jl_diffeq" => ["DP5", "Tsit5"],
    "scipy_ode" => ["vode", "lsoda", "dopri5", "dop853"],
)

PROBLEMS = [ScalarExpDecayProblem(), LinearOscillatorProblem(), OrbitEquationsProblem()]

# The enumeration of tests corresponds to the one from `test_ivp.py`.
@testset "Testing IVP interface from Julia" begin

    function fixture_self_prob(test_func)
        for impl in IMPLEMENTATIONS
            for prob in PROBLEMS
                self = IVP.Self(impl)
                test_func(self, prob)
            end
        end
    end

    function fixture_self(test_func)
        for impl in IMPLEMENTATIONS
            self = IVP.Self(impl)
            test_func(self)
        end
    end

    function fixture_impl_integrators_prob(test_func)
        for impl in IMPLEMENTATIONS
            integrators = INTEGRATORS[impl]
            prob = OrbitEquationsProblem()
            test_func(impl, integrators, prob)
        end
    end

    @testset "test_1__basic__should_work" begin
        fixture_self_prob() do self, prob
            @test IVP.set_initial_value(self, prob.y0, prob.t0) == 0
            @test self.N == length(prob.y0)
            @test IVP.set_rhs_fn(self, prob.rhs) == 0

            t1 = prob.t0 + 1
            times = collect(range(prob.t0, t1, 11))
            for t in times[2:end]
                IVP.integrate(self, t)
            end

            @test self.y ≈ prob.exact(t1) rtol=2e-4
        end
    end

    @testset "test_3__lower_tolerances__should_give_smaller_errors" begin
        fixture_self_prob() do self, prob
            IVP.set_initial_value(self, prob.y0, prob.t0)
            IVP.set_rhs_fn(self, prob.rhs)

            t1 = prob.t0 + 1.0
            times = collect(range(prob.t0, t1, 11))

            errors = []
            for tol in [1e-3, 1e-4, 1e-6, 1e-8]
                IVP.set_initial_value(self, prob.y0, prob.t0)
                IVP.set_tolerances(self, tol, tol)
                for t in times[2:end]
                    IVP.integrate(self, t)
                end
            end

            final_value = self.y
            true_value = prob.exact(t1)
            error = norm(final_value - true_value)
            push!(errors, error)

            for k in range(2, length(errors))
                @test errors[k-1] >= errors[k]
            end
        end
    end

    @testset "test_4__set_user_data__should_work" begin
        fixture_self() do impl_self
            params = (12, 2.7)
            p = IVPProblemWithUserData()
            IVP.set_initial_value(impl_self, p.y0, p.t0)
            IVP.set_user_data(impl_self, params)
            IVP.set_rhs_fn(impl_self, p.rhs)
            IVP.set_tolerances(impl_self, 1e-6, 1e-8)

            t1 = p.t0 + 1
            times = collect(range(p.t0, t1, 11))

            for t in times[2:end]
                IVP.integrate(impl_self, t)
            end

            @test impl_self.y ≈ p.exact(t1, params) rtol=1e-5 atol=1e-6
        end
    end

    @testset "test_5__check_that_we_can_set_integrator" begin
        fixture_impl_integrators_prob() do impl, integrators, prob
            dt = 0.125

            s = IVP.Self(impl)
            prob = OrbitEquationsProblem()

            IVP.set_initial_value(s, prob.y0, prob.t0)
            IVP.set_rhs_fn(s, prob.rhs)
            IVP.set_tolerances(s, 1e-2, 1e-2)

            IVP.integrate(s, prob.t0 + dt)
            value_1 = s.y

            println("Integrators list: ", integrators)
            for integrator_name in integrators
                println("Solver: ", impl)
                println("Integrator: ", integrator_name)
                IVP.set_integrator(s, integrator_name, Dict())
                IVP.set_initial_value(s, prob.y0, prob.t0)
                IVP.integrate(s, prob.t0 + dt)
                value_2 = s.y

                @test value_1 ≈ value_2 rtol=1e-1 atol=1e-1
            end
        end
    end

    # test_ivp.py has this test but I am not sure that it is a good idea
    # to throw an exception when the user calls `set_integrator`
    # before setting the right-hand side function.
    # @testset "test_6__check_that_set_integrator_works_only_after_setting_rhs" begin
    # end

    @testset "test_7__unknown_integrator_name__should_throw_exception" begin
        fixture_self() do self
            p = ScalarExpDecayProblem()
            IVP.set_initial_value(self, p.y0, p.t0)
            IVP.set_rhs_fn(self, p.rhs)

            @test_throws ErrorException IVP.set_integrator(self, "i-am-not-known-integrator", Dict())
        end
    end
end

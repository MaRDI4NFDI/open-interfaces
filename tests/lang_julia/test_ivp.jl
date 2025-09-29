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
            y0[1] * cos(omega * t) + +y0[2] * sin(omega * t) / omega,
            -y0[1] * omega * sin(omega * t) + +y0[2] * cos(omega * t),
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

    function rhs(_, y, ydot, (a, b)::Tuple{Int,Float64})
        ydot[1] = y[1] + a
        ydot[2] = b * y[2]
        return 0
    end

    function exact(t, (a, b)::Tuple{Int,Float64})
        return [a * (exp(t) - 1), exp(b * t)]
    end

    IVPProblem(t0, y0, rhs, exact)
end


function MildlyStiffODESystem()
    t0 = 0.0
    y0 = [1.0, 0.0]

    function rhs(t, y, ydot, __)
        ydot[1] = -16 * y[1] + 12 * y[2] + 16 * cos(t) - 13 * sin(t)
        ydot[2] = 12 * y[1] - 9 * y[2] - 11 * cos(t) + 9 * sin(t)
        return 0
    end

    function exact(t)
        return [cos(t), sin(t)]
    end

    IVPProblem(t0, y0, rhs, exact)
end


POTENTIAL_IMPLEMENTATIONS = ["sundials_cvode", "jl_diffeq", "scipy_ode"]
IMPLEMENTATIONS = []
for impl in POTENTIAL_IMPLEMENTATIONS
    implh = load_impl("ivp", impl, 1, 0)
    if implh == OpenInterfaces.OIF_BRIDGE_NOT_AVAILABLE_ERROR
        println(
            "Bridge component for implementation '",
            impl,
            "' is not available. Skipping the test",
        )
        continue
    end
    if implh == OpenInterfaces.OIF_IMPL_NOT_AVAILABLE_ERROR
        println("Implementation ", impl, " is not available. Skipping the test")
        continue
    end
    unload_impl(implh)
    push!(IMPLEMENTATIONS, impl)
end

INTEGRATORS = Dict()
if "sundials_cvode" in IMPLEMENTATIONS
    INTEGRATORS["sundials_cvode"] = ["adams", "bdf"]
end
if "jl_diffeq" in IMPLEMENTATIONS
    INTEGRATORS["jl_diffeq"] = ["DP5", "Tsit5"]
end
if "scipy_ode" in IMPLEMENTATIONS
    INTEGRATORS["scipy_ode"] = ["vode", "lsoda", "dopri5", "dop853"]
end

PROBLEMS = [ScalarExpDecayProblem(), LinearOscillatorProblem(), OrbitEquationsProblem()]

# The enumeration of tests corresponds to the one from `test_ivp.py`.
@testset verbose = true "Testing IVP interface from Julia" begin

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

            @test self.y ≈ prob.exact(t1) rtol = 2e-4
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

            @test impl_self.y ≈ p.exact(t1, params) rtol = 1e-5 atol = 1e-6
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

                @test value_1 ≈ value_2 rtol = 1e-1 atol = 1e-1
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

            @test_throws ErrorException IVP.set_integrator(
                self,
                "i-am-not-known-integrator",
                Dict(),
            )
        end
    end

    @testset "IVP, OIF_CONFIG_DICT tests" begin
        if "scipy_ode" in IMPLEMENTATIONS
            @testset "test_1__config_dict_scipy_ode__should_accept_alright" begin
                s = IVP.Self("scipy_ode")
                p = ScalarExpDecayProblem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                params = Dict("method" => "bdf", "order" => 1)
                @test IVP.set_integrator(s, "vode", params) == 0
            end
        else
            @test_skip println(
                "Implementation 'scipy_ode' not available. Skipping the test",
            )
        end

        if "jl_diffeq" in IMPLEMENTATIONS
            @testset "test_2__config_dict_with_bad_types__should_throw" begin
                s = IVP.Self("jl_diffeq")
                p = ScalarExpDecayProblem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                # Value 10^10 is too large to be represented as Int32
                # and the code should throw an exception.
                params = Dict("method" => 10^10, "order" => 1)
                @test_throws ErrorException IVP.set_integrator(s, "vode", params)
            end
        else
            @test_skip println(
                "Implementation 'jl_diffeq' not available. Skipping the test",
            )
        end

        if "scipy_ode" in IMPLEMENTATIONS
            @testset "test_3__malformed_config_dict_scipy_ode__should_throw" begin
                s = IVP.Self("scipy_ode")
                p = ScalarExpDecayProblem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)
                params = Dict("method" => "bdf", "wrong-param-name" => 1)
                @test_throws ErrorException IVP.set_integrator(s, "vode", params)
            end
        else
            @test_skip println(
                "Implementation 'scipy_ode' not available. Skipping the test",
            )
        end

        if "scipy_ode" in IMPLEMENTATIONS
            @testset "test_4__config_dict_scipy_ode__should_succeed_with_enough_nsteps" begin
                s = IVP.Self("scipy_ode")
                p = MildlyStiffODESystem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                t1 = p.t0 + 1

                # Set a very small number of steps, so that integrator fails.
                IVP.set_integrator(s, "dopri5", Dict("nsteps" => 1))
                @test_throws ErrorException IVP.integrate(s, t1)

                # Set large number of steps, so that integrator succeeds.
                IVP.set_integrator(s, "dopri5", Dict("nsteps" => 10_000))
                @test IVP.integrate(s, t1) == 0
            end
        else
            @test_skip println(
                "Implementation 'scipy_ode' not available. Skipping the test",
            )
        end

        if "sundials_cvode" in IMPLEMENTATIONS
            @testset "test_5__config_dict_cvode__fails_when_max_num_steps_too_small" begin
                s = IVP.Self("sundials_cvode")
                p = MildlyStiffODESystem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                IVP.set_integrator(s, "bdf", Dict("max_num_steps" => 50))

                t1 = p.t0 + 1

                @test_throws ErrorException s.integrate(t1)
            end
        else
            @test_skip println(
                "Implementation 'sundials_cvode' not available. Skipping the test",
            )
        end


        if "sundials_cvode" in IMPLEMENTATIONS
            @testset "test_6__config_dict_sundials_cvode__fails_with_false_options" begin
                s = IVP.Self("sundials_cvode")
                p = MildlyStiffODESystem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                # Should error on the unknown option.
                @test_throws ErrorException s.set_integrator(
                    "bdf",
                    Dict("max_num_steps_typo" => 50),
                )
            end
        else
            @test_skip println(
                "Implementation 'sundials_cvode' not available. Skipping the test",
            )
        end

        if "jl_diffeq" in IMPLEMENTATIONS
            @testset "test_7__config_dict_jl_diffeq__works" begin
                s = IVP.Self("jl_diffeq")
                p = MildlyStiffODESystem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                params = Dict("chunk_size" => 10, "autodiff" => false)
                @test IVP.set_integrator(s, "Rosenbrock23", params) == 0
            end
        else
            @test_skip println(
                "Implementation 'jl_diffeq' not available. Skipping the test",
            )
        end

        if "jl_diffeq" in IMPLEMENTATIONS
            @testset "test_8__config_dict_jl_diffeq__fails_when_unknown_options" begin
                s = IVP.Self("jl_diffeq")
                p = MildlyStiffODESystem()
                IVP.set_initial_value(s, p.y0, p.t0)
                IVP.set_rhs_fn(s, p.rhs)

                bad_params = Dict("unknown_option" => 10_000)
                @test_throws ErrorException IVP.set_integrator(s, "DP5", bad_params)
            end
        else
            @test_skip println(
                "Implementation 'jl_diffeq' not available. Skipping the test",
            )
        end
    end
end

using LinearAlgebra
using NonlinearSolve
using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP

# IMPLEMENTATIONS = ["c_lapack", "numpy", "jl_backslash"]
IMPLEMENTATIONS = ["sundials_cvode"]


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

        nonlin_prob = NonlinearProblem(f, [1.0], (p_eps,))
        sol = solve(nonlin_prob)
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


PROBLEMS = [ScalarExpDecayProblem(), LinearOscillatorProblem(), OrbitEquationsProblem()]

# The enumeration of tests corresponds to the one from `test_ivp.py`.
@testset "Testing IVP interface from Julia" begin

    function test(testCore)
        for impl in IMPLEMENTATIONS
            for p in PROBLEMS
                impl_self = IVP.Self(impl)
                testCore(impl_self, p)
            end
        end
    end

    function test_one_problem(testCore)
        for impl in IMPLEMENTATIONS
            impl_self = IVP.Self(impl)
            testCore(impl_self)
        end
    end

    @testset "test_1__basic__should_work" begin
        test() do impl_self, p
            @test IVP.set_initial_value(impl_self, p.y0, p.t0) == 0
            @test impl_self.N == length(p.y0)
            @test IVP.set_rhs_fn(impl_self, p.rhs) == 0

            t1 = p.t0 + 1
            times = collect(range(p.t0, t1, 11))
            for t in times[2:end]
                IVP.integrate(impl_self, t)
            end

            @test impl_self.y ≈ p.exact(t1) rtol=2e-4
        end
    end

    @testset "test_3__lower_tolerances__should_give_smaller_errors" begin
        test() do impl_self, p
            IVP.set_initial_value(impl_self, p.y0, p.t0)
            IVP.set_rhs_fn(impl_self, p.rhs)

            t1 = p.t0 + 1.0
            times = collect(range(p.t0, t1, 11))

            errors = []
            for tol in [1e-3, 1e-4, 1e-6, 1e-8]
                IVP.set_initial_value(impl_self, p.y0, p.t0)
                IVP.set_tolerances(impl_self, tol, tol)
                for t in times[2:end]
                    IVP.integrate(impl_self, t)
                end
            end

            final_value = impl_self.y
            true_value = p.exact(t1)
            error = norm(final_value - true_value)
            push!(errors, error)

            for k in range(2, length(errors))
                @test errors[k-1] >= errors[k]
            end
        end
    end

    @testset "test_4__set_user_data__should_work" begin
        test_one_problem() do impl_self
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
end

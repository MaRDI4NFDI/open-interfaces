using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP

# IMPLEMENTATIONS = ["c_lapack", "numpy", "jl_backslash"]
IMPLEMENTATIONS = ["sundials_cvode"]


mutable struct ScalarExpDecayProblem
    t0::Float64 = 0.0
    y0::Vector{Float64}
    rhs::Function
    exact::Function
    ScalarExpDecayProblem()
        t0 = 0.0
        y0 = [1.0]
        function rhs(t, y, ydot, user_data)
            ydot[1] = -y[1]
            return 0
        end
        exact = (t) -> y0[1] * exp(-t)

        new(t0, y0, rhs, exact)
    end
end


function rhs_fn(t, y, ydot, user_data)::Int32
    ydot[1], ydot[2] = -y[1], -y[2]  # Example RHS function
    return 0
end

@testset "Testing IVP interface from Julia" begin

    function test(testCore)
        for impl in IMPLEMENTATIONS
            for p in PROBLEMS
                println("Testing impl $impl")
                println("Test problem $p")
                impl_self = IVP.Self(impl)
                println("Testing implementation implh =", impl_self.implh)
                testCore(impl_self, p)
            end
        end
    end

    @testset "test_1" begin
        test() do impl_self, p
            @test IVP.set_initial_value(impl_self, p.y0, p.t0) == 0
            @test impl_self.N == length(p.y0)
            @test IVP.set_rhs_fn(impl_self, p.rhs) == 0

            t1 = p.t0 + 1
            times = collect(range(p.t0, t1, 11))
            for t in times[2:end]
                IVP.integrate(impl_self, t)
            end

            @test impl_self.y[end] ≈ p.exact(t1) rtol=2e-4
        end
    end

end

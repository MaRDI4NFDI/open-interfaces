using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP

# IMPLEMENTATIONS = ["c_lapack", "numpy", "jl_backslash"]
IMPLEMENTATIONS = ["sundials_cvode"]


function rhs_fn(t, y, ydot, user_data)::Int32
    ydot[1], ydot[2] = -y[1], -y[2]  # Example RHS function
    return 0
end

@testset "Testing IVP interface from Julia" begin

    function test(testCore)
        for impl in IMPLEMENTATIONS
            println("Testing impl $impl")
            impl_self = IVP.Self(impl)
            println("Testing implementation implh =", impl_self.implh)
            testCore(impl_self)
        end
    end

    @testset "test_1" begin
        test() do impl_self
            y0 = [1.0, 2.0]
            t0 = 0.0
            rtol = 1e-6
            atol = 1e-12
            user_data = "example_data"
            integrator_name = "RK4"
            integrator_params = Dict("step_size" => 0.1)

            @test IVP.set_initial_value(impl_self, y0, t0) == 0
            @test impl_self.N == length(y0)
            @test IVP.set_rhs_fn(impl_self, rhs_fn) == 0
            @test IVP.set_tolerances(impl_self, rtol, atol) == 0
            # IVP.set_user_data(impl_self, user_data)
            # IVP.set_integrator(impl_self, integrator_name, integrator_params)

            t_final = 1.0
            times = collect(range(0, t_final, 11))
            for t in times[2:end]
                IVP.integrate(impl_self, t_final)
            end

            @test impl_self.y ≈ [exp(-t_final), 2*exp(-t_final)] rtol=1e-6 atol=1e-8

            # IVP.print_stats(impl_self)
        end
    end

end

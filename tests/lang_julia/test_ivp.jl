using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.IVP

# IMPLEMENTATIONS = ["c_lapack", "numpy", "jl_backslash"]
IMPLEMENTATIONS = ["sundials_cvode"]

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
            rhs_fn = (t, y) -> [-y[1], -y[2]]  # Example RHS function
            rtol = 1e-6
            atol = 1e-12
            user_data = "example_data"
            integrator_name = "RK4"
            integrator_params = Dict("step_size" => 0.1)

            IVP.set_initial_value(impl_self, y0, t0)
            @test impl_self.N == length(y0)
            IVP.set_rhs_fn(impl_self, rhs_fn)
            IVP.set_tolerances(impl_self, rtol, atol)
            # IVP.set_user_data(impl_self, user_data)
            # IVP.set_integrator(impl_self, integrator_name, integrator_params)

            # t_final = 10.0
            # IVP.integrate(impl_self, t_final)

            # @test impl_self.y ≈ [exp(-t_final), exp(-t_final)] rtol=1e-6 atol=1e-6

            # IVP.print_stats(impl_self)
        end
    end

end

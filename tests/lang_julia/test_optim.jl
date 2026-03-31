using LinearAlgebra
using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.Optim

# -----------------------------------------------------------------------------
# Checking what implementations are available.
POTENTIAL_IMPLEMENTATIONS = ["scipy_optimize", "ipopt_jl"]
IMPLEMENTATIONS = []
for impl in POTENTIAL_IMPLEMENTATIONS
    implh = load_impl("optim", impl, 1, 0)
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
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Problems
function convex_objective_fn(x, __)
    return sum(x .^ 2)
end


function convex_objective_with_args_fn(x, args)
    return sum((x - args) .^ 2)
end


function rosenbrock_objective_fn(x, __)
    """The Rosenbrock function with additional arguments"""
    a, b = (0.5, 1.0)
    return sum(a * (x[2:end] - x[1:(end-1)] .^ 2.0) .^ 2.0 + (1 .- x[1:(end-1)]) .^ 2.0) + b
end
# -----------------------------------------------------------------------------


@testset verbose = true "Testing Optim interface from Julia" begin

    function test(testCore)
        for impl in IMPLEMENTATIONS
            self = Optim.Self(impl)
            testCore(self)
        end
    end

    @testset "test__simple_convex_problem__converges" begin
        test() do self
            x0 = [0.5, 0.6, 0.7]

            Optim.set_initial_guess(self, x0)
            Optim.set_objective_fn(self, convex_objective_fn)

            status, message = Optim.minimize(self)
            x = self.x

            @test status == 0
            @test typeof(message) == String && length(message) > 0
            @test length(x) == length(x0)
            @test sum(abs.(x)) < 1e-6
        end
    end

    @testset "test__rosenbrock_problem__converges" begin
        test() do self
            x0 = [3.14, 2.72, 42.0, 9.81, 8.31]

            Optim.set_initial_guess(self, x0)
            Optim.set_objective_fn(self, rosenbrock_objective_fn)

            status, message = Optim.minimize(self)
            x = self.x

            @test status == 0
            @test length(x) == length(x0)
            @test all(abs.(x .- 1.0) .< 1e-5)  # The solution is [1, 1, ..., 1].
        end
    end

    @testset "test__parameterized_convex_problem__converges" begin
        test() do self
            x0 = [0.5, 0.6, 0.7]
            user_data = [2.0, 3.0, -1.0]

            Optim.set_initial_guess(self, x0)
            Optim.set_user_data(self, user_data)
            Optim.set_objective_fn(self, convex_objective_with_args_fn)

            status, message = Optim.minimize(self)
            x = self.x

            @test status == 0
            @test length(x) == length(x0)
            @test all(abs.(x .- user_data) .< 1e-6)
        end
    end

    # -------------------------------------------------------------------------
    if "scipy_optimize" in IMPLEMENTATIONS
        @testset "scipy_minimize__rosenbrock_fn__converges_better_with_tighter_tol" begin
            self = Optim.Self("scipy_optimize")
            x0 = [0.5, 0.6, 0.7]
            x_expected = [1.0, 1.0, 1.0]
            user_data = (10, -1)

            Optim.set_initial_guess(self, x0)
            Optim.set_user_data(self, user_data)
            Optim.set_objective_fn(self, rosenbrock_objective_fn)

            Optim.set_method(self, "nelder-mead", Dict("xatol" => 1e-6))
            status, message = Optim.minimize(self)
            x_1 = copy(self.x)

            Optim.set_method(self, "nelder-mead", Dict("xatol" => 1e-8))
            status, message = Optim.minimize(self)
            x_2 = copy(self.x)

            @test norm(x_2 .- x_expected) < norm(x_1 .- x_expected)
        end
    else
        @test_skip println(
            "Implementation 'scipy_optimize' not available. Skipping the test",
        )
    end

    # -------------------------------------------------------------------------
    if "ipopt_jl" in IMPLEMENTATIONS
        @testset "ipopt_jl__rosenbrock_fn__converges_better_with_tighter_tol" begin
            self = Optim.Self("ipopt_jl")
            x0 = [0.5, 0.6, 0.7]
            x_expected = [1.0, 1.0, 1.0]
            user_data = (10, -1)

            Optim.set_initial_guess(self, x0)
            Optim.set_user_data(self, user_data)
            Optim.set_objective_fn(self, rosenbrock_objective_fn)

            Optim.set_method(self, "", Dict("tol" => 1e-1))
            status, message = Optim.minimize(self)
            x_1 = copy(self.x)

            Optim.set_method(self, "", Dict("tol" => 1e-4))
            status, message = Optim.minimize(self)
            x_2 = copy(self.x)

            @test norm(x_2 .- x_expected) < norm(x_1 .- x_expected)
        end
    else
        @test_skip println("Implementation 'ipopt_jl' not available. Skipping the test")
    end
    # -------------------------------------------------------------------------

    @testset "test__set_method_with__wrong_method_params__are_not_accepted" begin
        test() do self
            @test_throws ErrorException Optim.set_method(
                self,
                "nelder-mead",
                Dict("wrong_param" => 42),
            )
        end
    end

end

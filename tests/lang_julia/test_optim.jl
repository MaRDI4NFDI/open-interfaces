using LinearAlgebra
using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.Optim

IMPLEMENTATIONS = ["scipy_optimize"]


# -----------------------------------------------------------------------------
# Problems
function convex_objective_fn(x, __)
    return sum(x .^2 )
end


function convex_objective_with_args_fn(x, args)
    return sum((x - args) .^ 2)
end


function rosenbrock_objective_fn(x, __)
    """The Rosenbrock function with additional arguments"""
    a, b = (0.5, 1.0)
    return sum(a * (x[2:end] - x[1:end-1] .^ 2.0) .^ 2.0 + (1 .- x[1:end-1]) .^ 2.0) + b
end
# -----------------------------------------------------------------------------


@testset "Testing Optim interface from Julia" begin

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

    @testset "test__parameterized_convex_problem__converges_better_with_tigher_tolerance" begin
        test() do self
            x0 = [0.5, 0.6, 0.7]
            user_data = [2.0, 7.0, -1.0]

            Optim.set_initial_guess(self, x0)
            Optim.set_user_data(self, user_data)
            Optim.set_objective_fn(self, convex_objective_with_args_fn)

            Optim.set_method(self, "nelder-mead", Dict("xatol" => 1e-6))
            status, message = Optim.minimize(self)
            x_1 = copy(self.x)

            Optim.set_method(self, "nelder-mead", Dict("xatol" => 1e-8))
            status, message = Optim.minimize(self)
            x_2 = copy(self.x)

            @test norm(x_2 .- user_data) < norm(x_1 .- user_data)
        end
    end

    @testset "test__set_method_with__wrong_method_params__are_not_accepted" begin
        test() do self
            @test_throws ErrorException Optim.set_method(
                 self, "nelder-mead", Dict("wrong_param" =>  42)
            )
        end
    end

end

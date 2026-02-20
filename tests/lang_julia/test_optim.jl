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
    return sum(a * (x[2:end] - x[1:end-1] .^ 2.0) .^ 2.0 + (1 - x[1:end-1]) .^ 2.0) + b
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

end

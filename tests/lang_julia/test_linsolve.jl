using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.Linsolve

IMPLEMENTATIONS = ["c_lapack", "numpy"]
IMPLEMENTATIONS = ["c_lapack"]

@testset "Testing Linsolve interface from Julia" begin

    function test(testCore)
        for impl in IMPLEMENTATIONS
            implh = load_impl("linsolve", impl, 1, 0)
            testCore(implh)
            unload_impl(implh)
        end
    end

    @testset "test_1" begin
        test() do implh
            A = [[1.0  1.0]; [-3.0 1.0]]
            b = [6.0, 2.0]
            x = Linsolve.solve(implh, A, b)

            @test A * x ≈ b rtol=1e-15 atol=1e-15
        end
    end

end

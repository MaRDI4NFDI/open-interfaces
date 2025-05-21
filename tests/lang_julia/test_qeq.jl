using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.QEQ

IMPLEMENTATIONS = ["c_qeq_solver", "jl_qeq_solver", "py_qeq_solver"]

function test(testCore)
    for impl in IMPLEMENTATIONS
        implh = load_impl("qeq", impl, 1, 0)
        testCore(implh)
        unload_impl(implh)
    end
end

@testset "Testing qeq interface from Julia" begin
    @testset "test_1" begin
        test() do implh
            a, b, c = 1.0, 5.0, 4.0
            roots = QEQ.solve_qeq(implh, a, b, c)

            @test roots ≈ [-4.0, -1.0] rtol=1e-15 atol=1e-15
        end
    end

    @testset "test_2" begin
        test() do implh
            a, b, c = 1.0, -2.0, 1.0
            roots = QEQ.solve_qeq(implh, a, b, c)

            @test roots ≈ [1.0, 1.0] rtol=1e-15
        end
    end

    @testset "test correct extreme_roots: should give fiveteen digits" begin
        test() do implh
            a, b, c = 1.0, -20_000.0, 1.0
            x = QEQ.solve_qeq(implh, a, b, c)
            sort!(x)

            @test x ≈ [5.000000012500001e-05, 19999.999949999998] rtol=1e-15 atol=1e-15
        end
    end
end


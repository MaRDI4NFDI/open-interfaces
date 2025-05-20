using Test

using OpenInterfaces
using OpenInterfaces.Interfaces.QEQ

@testset "Testing qeq interface from Julia" begin
    @testset begin
        implh = load_impl("qeq", "jl_qeq_solver", 1, 0)
        a, b, c = 1.0, 5.0, 4.0
        roots = QEQ.solve_qeq(implh, a, b, c)

        @test roots ≈ [-4.0, -1.0] rtol=1e-15 atol=1e-15
        unload_impl(implh);
    end
end

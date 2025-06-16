using Test

@testset "Julia tests for Open Interfaces" begin
    include("test_qeq.jl")
    include("test_linsolve.jl")
    include("test_ivp.jl")
end

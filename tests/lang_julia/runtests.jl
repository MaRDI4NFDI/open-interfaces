using Test
using SafeTestsets

@testset "Julia tests for Open Interfaces" begin
    @safetestset "Julia tests for QEQ Open Interface" begin
        include("test_qeq.jl")
    end

    @safetestset "Julia tests for Linsolve Open Interface" begin
        include("test_linsolve.jl")
    end

    @safetestset "Julia tests for IVP Open Interface" begin
        include("test_ivp.jl")
    end

    @safetestset "Julia tests for IVP Open Interface" begin
        include("test_examples.jl")
    end
end

using OpenInterfaces
using Test

print("TEST RUN")

@testset "OpenInterfaces.jl tests" begin
    include("test_solve.jl")
    include("test_expression.jl")
end

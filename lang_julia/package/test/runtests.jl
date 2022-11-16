using OpenInterfaces
using Test

print("TEST RUN")

@testset "OpenInterfaces.jl tests" begin
    include("test_all.jl")
end

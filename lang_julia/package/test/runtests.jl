using OpenInterfaces
using Test


if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", false)
    println("Tests disabled in automerge due to missing OpenInterfaces libraries")
else
  @testset "OpenInterfaces.jl tests" begin
      include("test_solve.jl")
      include("test_expression.jl")
  end
end

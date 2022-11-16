#!/usr/bin/env julia

using OpenInterfaces

if size(ARGS)[1] == 2
    const lang = ARGS[1]
    const expression = ARGS[2]
else
    const lang = "r"
    const expression = "print(7*6)"
end

oif = OpenInterfaces.init(lang)
const N = 2
A = [2., 0., 1., 0.];
b = [1., 1.]
x = [0.,0.]
ok = @ccall oif.oif_connector_solve(N::Cint, A::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
check_call(ret)

OpenInterfaces.init(oif)

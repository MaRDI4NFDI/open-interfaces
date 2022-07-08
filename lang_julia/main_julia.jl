#!/usr/bin/env julia

if size(ARGS)[1] == 2
    const lang = ARGS[1]
    const expression = ARGS[2]
else
    const lang = "r"
    const expression = "print(7*6)"
end
const oif = "liboif_connector"
ok = @ccall oif.oif_connector_init(lang::Cstring)::Cint
if ok != 0
    println("Cannot load connector")
    ccall(:jl_exit, Cvoid, (Int32,), ok)
end

@ccall oif.oif_connector_eval_expression(expression::Cstring)::Cint

const N = 2
A = [2., 0., 1., 0.];
b = [1., 1.]
x = [0.,0.]
ok = @ccall oif.oif_connector_solve(N::Cint, A::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
if ok != 0
    println("Cannot load connector")
    ccall(:jl_exit, Cvoid, (Int32,), ok)
end

@ccall oif.oif_connector_deinit()::Cvoid

#!/usr/bin/env julia

function check_call(result, msg="Cannot load connector")
    if result != 0
        println(msg)
        ccall(:jl_exit, Cvoid, (Int32,), ok)
    end
end

if size(ARGS)[1] == 2
    const lang = ARGS[1]
    const expression = ARGS[2]
else
    const lang = "r"
    const expression = "print(7*6)"
end
const oif = "./oif_connector/liboif_connector"
ret = @ccall oif.oif_connector_init(lang::Cstring)::Cint
check_call(ret)

ret = @ccall oif.oif_connector_eval_expression(expression::Cstring)::Cint
check_call(ret)

const N = 2
A = [2., 0., 1., 0.];
b = [1., 1.]
x = [0.,0.]
ok = @ccall oif.oif_connector_solve(N::Cint, A::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
check_call(ret)

@ccall oif.oif_connector_deinit()::Cvoid

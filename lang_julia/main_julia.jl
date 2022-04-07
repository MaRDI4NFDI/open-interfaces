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
@ccall oif.oif_connector_deinit()::Cvoid

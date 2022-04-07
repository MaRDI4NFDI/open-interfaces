#!/usr/bin/env julia

const oif = "liboif_connector"
const lang = "julia"
const expression = "print(7*6)"
ok = @ccall oif.oif_connector_init(lang::Cstring)::Cint
if ok != 0
    println("Cannot load connector")
    ccall(:jl_exit, Cvoid, (Int32,), ok)
end
@ccall oif.oif_connector_eval_expression(expression::Cstring)::Cint
@ccall oif.oif_connector_deinit()::Cvoid

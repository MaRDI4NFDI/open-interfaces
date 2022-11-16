module OpenInterfaces

const oif = "liboif_connector"

function check_call(result, msg="Cannot load connector")
    if result != 0
        flush(stdout)
        flush(stderr)
        error(msg)
    end
end

function init(lang)
    ret = @ccall "liboif_connector".oif_connector_init(lang::Cstring)::Cint
    check_call(ret, "cannot load connector for $lang")
end

function eval(expression)
    ret = @ccall "liboif_connector".oif_connector_eval_expression(expression::Cstring)::Cint
    check_call(ret, "cannot eval $expression")
end

function solve(N, A, b, x)
    ret = @ccall oif.oif_connector_solve(N::Cint, A::Ptr{Cdouble}, b::Ptr{Cdouble}, x::Ptr{Cdouble})::Cint
    check_call(ret, "cannot solve linear system of size $N")
end

function deinit()
    ret = @ccall "liboif_connector".oif_connector_deinit()::Cint
    check_call(ret, "failed to deinit connector")
end

end

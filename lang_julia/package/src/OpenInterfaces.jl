module OpenInterfaces

const oif = "liboif_connector"

function check_call(result, msg="Cannot load connector")
    if result != 0
        println(msg)
        ccall(:jl_exit, Cvoid, (Int32,), ok)
    end
end

function init(lang)
    ret = @ccall "liboif_connector".oif_connector_init("lang"::Cstring)::Cint
    check_call(ret)
    return oif
end

function eval(expression)
    ret = @ccall "liboif_connector".oif_connector_eval_expression(expression::Cstring)::Cint
    check_call(ret)
end

function deinit(oif)
    return @ccall "liboif_connector".oif_connector_deinit()::Cvoid
end

end

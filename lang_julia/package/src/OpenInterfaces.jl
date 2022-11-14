module OpenInterfaces

function check_call(result, msg="Cannot load connector")
    if result != 0
        println(msg)
        ccall(:jl_exit, Cvoid, (Int32,), ok)
    end
end

function init()
    oif = "liboif_connector"
    ret = @ccall oif.oif_connector_init(lang::Cstring)::Cint
    check_call(ret)

    ret = @ccall oif.oif_connector_eval_expression(expression::Cstring)::Cint
    check_call(ret)
    return oif
end

function deinit(oif)
    return @ccall oif.oif_connector_deinit()::Cvoid
end

end

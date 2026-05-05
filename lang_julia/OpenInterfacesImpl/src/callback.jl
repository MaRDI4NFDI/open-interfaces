module CallbackWrapper
export make_wrapper_over_c_callback

using OpenInterfaces

function make_wrapper_over_c_callback(fn_c::Ptr{Cvoid}, oif_argtypes, oif_restype)::Function
    c_argtypes = Tuple(
        map(argtype -> begin
                if argtype == OIF_TYPE_I32
                    Cint
                elseif argtype == OIF_TYPE_F64
                    Cdouble
                elseif argtype == OIF_TYPE_ARRAY_F64
                    Ptr{OIFArrayF64}
                elseif argtype == OIF_TYPE_USER_DATA
                    Ptr{Cvoid}
                else
                    error("Unsupported argument type: $argtype")
                end
            end, oif_argtypes)
    )

    # Build a concrete Tuple type e.g. Tuple{Cdouble,Cdouble}.
    # This IS a valid type parameter, unlike a plain tuple value (Cdouble, Cdouble).
    c_argtypes_type = Core.apply_type(Tuple, c_argtypes...)

    c_restype =
        if oif_restype == OIF_TYPE_I32 || oif_restype == OIF_TYPE_INT
            # The second condition is just for explicitness, as `INT` is an alias to `I32`.
            Cint
        elseif oif_restype == OIF_TYPE_F64
            Cdouble
        else
            error(
                "Unsupported return type: $oif_restype. Only supported types are Cint and Cdouble",
            )
        end

    function wrapper(args...)
        c_args_from_jl_args = Tuple(
            map((argtype, argvalue) -> begin
                    if argtype == OIF_TYPE_I32
                        argvalue
                    elseif argtype == OIF_TYPE_F64
                        argvalue
                    elseif argtype == OIF_TYPE_ARRAY_F64
                        Ref(OIFArrayF64(argvalue))
                    elseif argtype == OIF_TYPE_USER_DATA
                        argvalue === nothing ? C_NULL : argvalue
                    else
                        error("Unsupported argument type: $argtype")
                    end
                end, oif_argtypes, args)
        )

        return @GC.preserve args call_c(fn_c, c_argtypes_type, c_restype, c_args_from_jl_args)
    end

    return wrapper
end

@generated function call_c(fn_p, ::Type{argtypes}, ::Type{restype}, args) where {argtypes, restype}

    n = length(argtypes.parameters)

    argtypes_expr = Expr(:tuple, map(t -> :( $t ), argtypes.parameters)...)
    # Generate :(args[1]), :(args[2]), … at code-generation time so that ccall
    # sees a fixed, statically-known argument list (runtime splatting is not allowed).
    args_expr = [:(args[$i]) for i in 1:n]

    quote
        ccall(fn_p, $restype, $argtypes_expr, $(args_expr...))
    end
end

end

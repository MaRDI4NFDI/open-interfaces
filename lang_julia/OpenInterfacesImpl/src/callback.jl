module CallbackWrapper
export make_wrapper_over_c_callback

import SciMLBase

using OpenInterfaces: OIFArrayF64, OIF_ARRAY_C_CONTIGUOUS, OIF_ARRAY_F_CONTIGUOUS

function make_wrapper_over_c_callback(fn_c::Ptr{Cvoid})::Function
    function wrapper(t::Float64, y::Vector{Float64}, ydot::Vector{Float64}, user_data)::Int
        if typeof(user_data) == SciMLBase.NullParameters
            user_data = C_NULL
        end

        oif_y = Ref(OIFArrayF64(y))
        oif_ydot = Ref(OIFArrayF64(ydot))

        @ccall $fn_c(
            t::Float64,
            oif_y::Ptr{OIFArrayF64},
            oif_ydot::Ptr{OIFArrayF64},
            user_data::Ptr{Cvoid},
        )::Cint

        return 0
    end
    return wrapper
end

end

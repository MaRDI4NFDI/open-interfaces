module CallbackWrapper
export make_wrapper_for_c_callback

import SciMLBase

using OpenInterfaces: OIFArrayF64

function make_wrapper_over_c_callback(fn_c::Ptr{Cvoid})::Function
    function wrapper(t, y, ydot, user_data)::Int
        if typeof(user_data) == SciMLBase.NullParameters
            user_data = C_NULL
        end
        dimensions = Base.unsafe_convert(Ptr{Clong}, collect(size(y)))
        data = Base.unsafe_convert(Ptr{Float64}, y)
        oif_y = Ref(OIFArrayF64(ndims(y), dimensions, data))

        dimensions = Base.unsafe_convert(Ptr{Clong}, collect(size(ydot)))
        data = Base.unsafe_convert(Ptr{Float64}, ydot)
        oif_ydot = Ref(OIFArrayF64(ndims(ydot), dimensions, data))

        @ccall $fn_c(t::Float64, oif_y::Ptr{OIFArrayF64}, oif_ydot::Ptr{OIFArrayF64}, user_data::Ptr{Cvoid})::Cint
        return 0
    end
    return wrapper
end
end

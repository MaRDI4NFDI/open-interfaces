module CallbackWrapper
export make_wrapper_for_c_callback

using OpenInterfaces: OIFArrayF64

function make_wrapper_over_c_callback(fn_c::Ptr{Cvoid})::Function
    function wrapper(t, y, ydot, user_data)::Int
        @ccall $fn_c(t::Float64, y::Ptr{OIFArrayF64}, ydot::Ptr{OIFArrayF64}, user_data::Ptr{Cvoid})::Cint
        return 0
    end
    return wrapper
end
end

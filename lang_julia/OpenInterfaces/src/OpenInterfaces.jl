module OpenInterfaces
# Define constants and data structures equivalent to the C interface.
# See oif/api.h

# Type ids
export OIF_INT, OIF_FLOAT64, OIF_ARRAY_F64, OIF_STR, OIF_CALLBACK, OIF_USER_DATA, OIF_CONFIG_DICT

# Language ids
export OIF_LANG_C, OIF_LANG_CXX, OIF_LANG_PYTHON, OIF_LANG_JULIA, OIF_LANG_R

# Error codes
export OIF_ERROR, OIF_IMPL_INIT_ERROR

# Data structures
export OIFArgs, OIFArrayF64, OIFCallback, OIFUserData

@enum OIFArgType begin
    OIF_INT = 1
    # OIF_FLOAT32 = 2
    OIF_FLOAT64 = 3
    # OIF_FLOAT32_P = 4
    OIF_ARRAY_F64 = 5
    OIF_STR = 6
    OIF_CALLBACK = 7
    OIF_USER_DATA = 8
    OIF_CONFIG_DICT = 9
end

@enum OIFLang begin
    OIF_LANG_C = 1
    OIF_LANG_CXX = 2
    OIF_LANG_PYTHON = 3
    OIF_LANG_JULIA = 4
    OIF_LANG_R = 5
    OIF_LANG_COUNT = 6
end

@enum OIFError begin
    OIF_ERROR = -1
    OIF_IMPL_INIT_ERROR = -2
end

struct OIFArgs
    num_args::Int64
    arg_types::Ptr{OIFArgType}
    arg_values::Ptr{Ptr{Cvoid}}
end

struct OIFArrayF64
    nd::Int32
    dimensions::Ptr{Int64}
    data::Ptr{Float64}
end

struct OIFCallback
    src::Int32
    fn_p_py::Ptr{Cvoid}
    fn_p_c::Ptr{Cvoid}
end

struct OIFUserData
    src::Int32
    c::Ptr{Cvoid}
    py::Ptr{Cvoid}
end

end # module OpenInterfaces

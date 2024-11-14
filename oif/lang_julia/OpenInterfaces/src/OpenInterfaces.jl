module OpenInterfaces
# Define constants and data structures equivalent to the C interface.
# See oif/api.h

export OIFArgType, OIFArgs, OIFArrayF64
export OIF_INT, OIF_FLOAT64, OIF_ARRAY_F64, OIF_STR, OIF_CALLBACK, OIF_USER_DATA, OIF_CONFIG_DICT

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

struct OIFArrayF64
    nd::Int32
    dimensions::Ptr{Int64}
    data::Ptr{Float64}
end

struct OIFArgs
    num_args::Int64
    arg_types::Ptr{OIFArgType}
    arg_values::Ptr{Ptr{Cvoid}}
end

end # module OpenInterfaces

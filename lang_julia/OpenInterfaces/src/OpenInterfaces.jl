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

export load_impl, call_impl, unload_impl

using Libdl

const ImplHandle = Int32

const OIFArgType = Int32

const OIF_INT::OIFArgType = 1
# const OIF_FLOAT32::OIFArgType = 2
const OIF_FLOAT64::OIFArgType = 3
# const OIF_FLOAT32_P::OIFArgType = 4
const OIF_ARRAY_F64::OIFArgType = 5
const OIF_STR::OIFArgType = 6
const OIF_CALLBACK::OIFArgType = 7
const OIF_USER_DATA::OIFArgType = 8
const OIF_CONFIG_DICT::OIFArgType = 9

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

const OIF_ARRAY_C_CONTIGUOUS::Int32 = 0x0001
const OIF_ARRAY_F_CONTIGUOUS::Int32 = 0x0002

struct OIFArgs
    num_args::Int64
    arg_types::Ptr{OIFArgType}
    arg_values::Ptr{Ptr{Cvoid}}
end

struct OIFArrayF64
    nd::Int32
    dimensions::Ptr{Int64}
    data::Ptr{Float64}
    flags::Int32
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

const lib_dispatch = Ref{Ptr{Cvoid}}(0)
const load_interface_impl_fn = Ref{Ptr{Cvoid}}(0)
const call_interface_impl_fn = Ref{Ptr{Cvoid}}(0)
const unload_interface_impl_fn = Ref{Ptr{Cvoid}}(0)

export _lib_dispatch

function __init__()
    lib_dispatch[] = Libdl.dlopen("liboif_dispatch.so")
    load_interface_impl_fn[] = dlsym(lib_dispatch[], :load_interface_impl)
    call_interface_impl_fn[] = dlsym(lib_dispatch[], :call_interface_impl)
    unload_interface_impl_fn[] = dlsym(lib_dispatch[], :unload_interface_impl)
end

function load_impl(interface::String, impl::String, major::Int, minor::Int)::ImplHandle
    implh = @ccall $(load_interface_impl_fn[])(
        interface::Cstring,
        impl::Cstring,
        major::UInt,
        minor::UInt
    )::Int

    if implh < 0
        error("Failed to load interface implementation: $impl")
    end

    return implh
end

function call_impl(implh::ImplHandle, func_name::String, in_user_args::Tuple{Vararg{Any}}, out_user_args::Tuple{Vararg{Any}})::Int
    in_num_args = length(in_user_args)
    out_num_args = length(out_user_args)

    temp_refs = Vector{Any}()

    # Allocate memory for the argument types and values
    in_arg_types = Vector{OIFArgType}(undef, in_num_args)
    in_arg_values = Vector{Ptr{Cvoid}}(undef, in_num_args)
    out_arg_types = Vector{OIFArgType}(undef, out_num_args)
    out_arg_values = Vector{Ptr{Cvoid}}(undef, out_num_args)

    for (i, arg) in enumerate(in_user_args)
        if typeof(arg) == Int
            in_arg_types[i] = OIF_INT
            in_arg_values[i] = Ref(arg)
        elseif typeof(arg) == Float64
            in_arg_types[i] = OIF_FLOAT64
            arg_ref = Ref(arg)
            push!(temp_refs, arg_ref)
            in_arg_values[i] = Base.unsafe_convert(Ptr{Cvoid}, arg_ref)
        elseif typeof(arg) <: AbstractArray{Float64}
            nd = ndims(arg)

            dims_jl = collect(size(arg))
            push!(temp_refs, dims_jl)
            dimensions = Base.unsafe_convert(Ptr{Clong}, dims_jl)
            push!(temp_refs, dimensions)
            data = pointer(arg)
            push!(temp_refs, data)

            if nd == 1
                # One-dimensional arrays are both C and Fortran-contiguous.
                flags = OIF_ARRAY_C_CONTIGUOUS | OIF_ARRAY_F_CONTIGUOUS
            else
                # Julia arrays are Fortran-contiguous by default.
                flags = OIF_ARRAY_F_CONTIGUOUS
            end

            arr = OIFArrayF64(nd, dimensions, data, flags)
            push!(temp_refs, arr)

            arr_ref = Ref(arr)
            push!(temp_refs, arr_ref)
            arr_p = Base.unsafe_convert(Ptr{Cvoid}, arr_ref)
            push!(temp_refs, arr_p)

            arr_p_ref = Ref(arr_p)
            push!(temp_refs, arr_p_ref)
            arr_p_p = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, arr_p_ref)

            in_arg_types[i] = OIF_ARRAY_F64
            in_arg_values[i] = arr_p_p
        elseif typeof(arg) == String
            in_arg_types[i] = OIF_STR
            in_arg_values[i] = pointer(arg)
        elseif typeof(arg) == OIFCallback
            in_arg_types[i] = OIF_CALLBACK
            in_arg_values[i] = pointer(arg)
        elseif typeof(arg) == OIFUserData
            in_arg_types[i] = OIF_USER_DATA
            in_arg_values[i] = pointer(arg)
        elseif typeof(arg) == Dict
            in_arg_types[i] = OIF_CONFIG_DICT
            # Convert the dictionary to a pointer
            dict_ptr = pointer(arg)
            in_arg_values[i] = dict_ptr
        else
            error("Cannot convert input argument $(arg) of type $(typeof(arg))")
        end
    end

    for (i, arg) in enumerate(out_user_args)
        if typeof(arg) == Int
            out_arg_types[i] = OIF_INT
            out_arg_values[i] = pointer(arg)
        elseif typeof(arg) == Float64
            out_arg_types[i] = OIF_FLOAT64
            out_arg_values[i] = pointer(arg)
        elseif typeof(arg) <: AbstractArray{Float64}
            nd = ndims(arg)
            dims_jl = collect(size(arg))
            push!(temp_refs, dims_jl)
            dimensions = Base.unsafe_convert(Ptr{Clong}, dims_jl)
            push!(temp_refs, dimensions)
            data = pointer(arg)
            push!(temp_refs, data)

            if nd == 1
                # One-dimensional arrays are both C and Fortran-contiguous.
                flags = OIF_ARRAY_C_CONTIGUOUS | OIF_ARRAY_F_CONTIGUOUS
            else
                # Julia arrays are Fortran-contiguous by default.
                flags = OIF_ARRAY_F_CONTIGUOUS
            end

            arr = OIFArrayF64(nd, dimensions, data, flags)
            push!(temp_refs, arr)

            arr_ref = Ref(arr)
            push!(temp_refs, arr_ref)
            arr_p = Base.unsafe_convert(Ptr{Cvoid}, arr_ref)
            push!(temp_refs, arr_p)

            arr_p_ref = Ref(arr_p)
            push!(temp_refs, arr_p_ref)
            arr_p_p = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, arr_p_ref)

            out_arg_types[i] = OIF_ARRAY_F64
            out_arg_values[i] = arr_p_p
        else
            error("Cannot convert output argument $(arg) of type $(typeof(arg))")
        end
    end

    in_args = Ref(OIFArgs(in_num_args, pointer(in_arg_types), pointer(in_arg_values)))
    out_args = Ref(OIFArgs(out_num_args, pointer(out_arg_types), pointer(out_arg_values)))

    result = GC.@preserve in_arg_types in_arg_values out_arg_types out_arg_values begin
        @ccall $(call_interface_impl_fn[])(
            implh::Int,
            func_name::Cstring,
            in_args::Ptr{OIFArgs},
            out_args::Ptr{OIFArgs}
        )::Int
    end

    return result
end

function unload_impl(implh::ImplHandle)
    status = @ccall $(unload_interface_impl_fn[])(
        implh::Cint
    )::Int

    if status != 0
        error("Failed to unload interface implementation with id '$impl'")
    end
end

include("interfaces.jl")

end # module OpenInterfaces

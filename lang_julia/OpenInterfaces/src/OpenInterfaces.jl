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
export OIFArgType, OIFArgs, OIFArrayF64, OIFCallback, OIFUserData

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

const OIF_LANG_C::Int32 = 1
const OIF_LANG_CXX::Int32 = 2
const OIF_LANG_PYTHON::Int32 = 3
const OIF_LANG_JULIA::Int32 = 4
const OIF_LANG_R::Int32 = 5
const OIF_LANG_COUNT::Int32 = 6

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
    fn_p_c::Ptr{Cvoid}
    fn_p_jl::Ptr{Cvoid}
    fn_p_py::Ptr{Cvoid}
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
            arg_ref = Ref(arg)
            push!(temp_refs, arg_ref)
            in_arg_types[i] = OIF_CALLBACK
            in_arg_values[i] = Base.unsafe_convert(Ptr{Cvoid}, arg_ref)
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

function make_oif_callback(fn, argtypes::NTuple{N, OIFArgType}, restype::OIFArgType)::OIFCallback where {N}
    @assert restype == OIF_INT "Only OIF_INT is supported as a return type for callbacks because 
        it is used to indicate success or failure in a C-compatible way."
    c_argtypes::Array{Any, 1} = []

    for argtype in argtypes
        if argtype == OIF_INT
            push!(c_argtypes, Cint)
        elseif argtype == OIF_FLOAT64
            push!(c_argtypes, Cdouble)
        elseif argtype == OIF_ARRAY_F64
            push!(c_argtypes, Ptr{OIFArrayF64})
        elseif argtype == OIF_USER_DATA
            push!(c_argtypes, Ptr{Cvoid})
        else
            error("Unsupported argument type: $argtype")
        end
    end

    fn_wrapper = _make_c_func_wrapper_over_jl_fn(fn, argtypes, restype)

    # Build the function wrapper in C.
    # I have to use `@cfunction` to create a C function pointer that can be called from C code
    # However, the variable `c_argtypes` can be known only at runtime,
    # not at compile time.
    # Therefore, I get a compilation error,
    # that `c_argtypes` must be literal tuple.
    # All this black magic with eval is here to overcome this limitation.
    c_argtypes_expr = Expr(:tuple, (map(t->QuoteNode(t), c_argtypes))...)
    cfunction_expr = :(@cfunction($fn_wrapper, Cint, $c_argtypes_expr))

    fn_p_c = eval(cfunction_expr)
    # Convert the Julia function to a pointer.
    fn_p_jl = Base.unsafe_convert(Ptr{Cvoid}, fn)
    # Python pointer should be null.
    fn_p_py = C_NULL

    return OIFCallback(OIF_LANG_JULIA, fn_p_c, fn_p_jl, fn_p_py)
end


function _make_c_func_wrapper_over_jl_fn(fn, argtypes::NTuple{N, OIFArgType}, restype::Int32) where {N}
    # This function creates a C function wrapper that calls the Julia function.
    # The actual implementation will depend on how you want to handle the arguments and return value.
    oif_argtypes = argtypes
    function wrapper(oif_args...)
        jl_args = Array{Any}(undef, 0)

        for (i, arg) in enumerate(oif_args)
            if oif_argtypes[i] == OIF_INT
                push!(jl_args, Int(arg))
            elseif oif_argtypes[i] == OIF_FLOAT64
                push!(jl_args, Float64(arg))
            elseif oif_argtypes[i] == OIF_ARRAY_F64
                # Convert the pointer to an array.
                oif_arr = unsafe_load(arg)
                dimensions = unsafe_load(oif_arr.dimensions)
                arr = unsafe_wrap(Array{Float64}, oif_arr.data, dimensions, own=false)

                push!(jl_args, arr)
            elseif oif_argtypes[i] == OIF_USER_DATA
                # Convert the pointer to a user data structure.
                user_data = unsafe_load(arg)
                push!(jl_args, user_data)
            else
                error("Unsupported argument type: $(oif_argtypes[i])")
            end
        end

        # Call the Julia function with the converted arguments.
        result::Int32 = fn(jl_args...)

        if result == nothing
            return Int32(0)
        end

        return result
    end

    return wrapper
end

include("interfaces.jl")

end # module OpenInterfaces

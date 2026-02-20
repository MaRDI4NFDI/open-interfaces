module Optim

using OpenInterfaces:
    ImplHandle,
    load_impl,
    call_impl,
    unload_impl,
    make_oif_callback,
    make_oif_user_data,
    OIF_TYPE_F64,
    OIF_TYPE_ARRAY_F64,
    OIF_TYPE_I32,
    OIF_TYPE_USER_DATA,
    OIF_TYPE_STRING,
    OIFUserData

export Self,
    set_initial_value,
    set_rhs_fn,
    set_tolerances,
    set_user_data,
    set_integrator,
    integrate,
    print_stats

mutable struct Self
    implh::ImplHandle
    N::Int32
    x0::Vector{Float64}
    x::Vector{Float64}
    objective_fn_wrapper::Any
    user_data::Ref{Any}
    oif_user_data::Any
    function Self(impl::String)
        implh = load_impl("optim", impl, 1, 0)
        self = new(ImplHandle(implh), 0, Float64[], Float64[], Nothing, Nothing, Nothing)
        finalizer(finalizing, self)
    end

    function finalizing(self::Self)
        """Finalization function to clean up resources."""
        unload_impl(self.implh)
    end
end

function set_initial_guess(self::Self, x0::Vector{Float64})
    """Set initial guess for the optimization problem"""
    self.N = length(x0)
    self.x0 = copy(x0)
    self.x = Vector{Float64}(undef, self.N)
    call_impl(self.implh, "set_initial_guess", (self.x0,), ())
end

function set_objective_fn(self::Self, objective_fn::Function)
    """Specify right-hand side function f."""
    self.objective_fn_wrapper = make_oif_callback(
        objective_fn, (OIF_TYPE_ARRAY_F64, OIF_TYPE_USER_DATA), OIF_TYPE_F64,
    )
    call_impl(self.implh, "set_objective_fn", (self.objective_fn_wrapper,), ())
end

# function set_tolerances(self::Self, rtol::Float64, atol::Float64)
#     """Specify relative and absolute tolerances, respectively."""
#     call_impl(self.implh, "set_tolerances", (rtol, atol), ())
# end

function set_user_data(self::Self, user_data::Any)
    """Specify additional data that will be used for right-hand side function."""
    self.user_data = user_data
    self.oif_user_data = make_oif_user_data(self.user_data)
    call_impl(self.implh, "set_user_data", (self.oif_user_data,), ())
end

function set_method(self::Self, method_name::String, method_params::Dict)
    """Set integrator, if the name is recognizable."""
    call_impl(self.implh, "set_method", (method_name, method_params), ())
end

function minimize(self::Self)::Tuple{Int, String}
    """Solve minimization problem"""
    status::Int, message::String = call_impl(self.implh, "minimize", (), (self.x,), (OIF_TYPE_I32, OIF_TYPE_STRING),
             )
    return status, message
end

end  # module Optim

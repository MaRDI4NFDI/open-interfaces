module IVP

using OpenInterfaces: ImplHandle, load_impl, call_impl, unload_impl, make_oif_callback, make_oif_user_data, OIF_FLOAT64, OIF_ARRAY_F64, OIF_INT, OIF_USER_DATA, OIFUserData

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
    y::Vector{Float64}
    rhs_fn_wrapper
    user_data_ref::Ref{Any}
    oif_user_data
    function Self(impl::String)
        implh = load_impl("ivp", impl, 1, 0)
        self = new(ImplHandle(implh), 0, Float64[], Nothing, Nothing, Nothing)
        finalizer(finalizing, self)
    end

    function finalizing(self::Self)
        """Finalization function to clean up resources."""
        unload_impl(self.implh)
    end
end

function set_initial_value(self::Self, y0::Vector{Float64}, t0::Float64)
    """Set initial value y(t0) = y0."""
    self.N = length(y0)
    self.y = Vector{Float64}(undef, self.N)
    call_impl(self.implh, "set_initial_value", (y0, t0), ())
end

function set_rhs_fn(self::Self, rhs_fn::Function)
    """Specify right-hand side function f."""
    self.rhs_fn_wrapper = make_oif_callback(
        rhs_fn,
        (OIF_FLOAT64, OIF_ARRAY_F64, OIF_ARRAY_F64, OIF_USER_DATA),
        OIF_INT,
    )
    call_impl(self.implh, "set_rhs_fn", (self.rhs_fn_wrapper,), ())
end

function set_tolerances(self::Self, rtol::Float64, atol::Float64)
    """Specify relative and absolute tolerances, respectively."""
    call_impl(self.implh, "set_tolerances", (rtol, atol), ())
end

function set_user_data(self::Self, user_data::Any)
    """Specify additional data that will be used for right-hand side function."""
    self.user_data_ref = user_data
    self.oif_user_data = make_oif_user_data(self.user_data_ref)
    call_impl(self.implh, "set_user_data", (self.oif_user_data,), ())
end

function set_integrator(self::Self, integrator_name::String, integrator_params::Dict)
    """Set integrator, if the name is recognizable."""
    call_impl(self.implh, "set_integrator", (integrator_name, integrator_params), ())
end

function integrate(self::Self, t::Float64)
    """Integrate to time `t` and write solution to `y`."""
    call_impl(self.implh, "integrate", (t,), (self.y,))
end

function print_stats(self::Self)
    """Print integration statistics."""
    println("Printing integration statistics.")
end

end

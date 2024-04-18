module JlDiffEq
export set_initial_value, set_rhs_fn, set_tolerances, integrate, set_user_data, callback, OIFArrayF64

function set_initial_value(y0, t0)
    println("I am setting initial value")
    return 0
end

function set_rhs_fn(rhs)::Int
    println("I am setting rhs") 
    return 0
end

function set_tolerances(rtol::Float64, atol::Float64)::Int
    println("I am setting tolerances")
    return 0
end

function integrate!(t::Float64, y::Vector{Float64})::Int
    println("I am integrating")
    return 0
end

function set_user_data(user_data)::Int
    println("I am setting user data")
end

end

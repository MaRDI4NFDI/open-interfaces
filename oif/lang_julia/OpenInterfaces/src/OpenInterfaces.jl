module OpenInterfaces

export OIFArrayF64

struct OIFArrayF64
    nd::Int32
    dimensions::Ptr{Int64}
    data::Ptr{Float64}
end

end # module OpenInterfaces

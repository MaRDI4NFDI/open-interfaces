module OIFSerialization
export deserialize
import MsgPack

function deserialize(sd::Ptr)::Dict
    # We get unsigned bytes and it is important to keep them this way
    # to get correct conversion.
    sd_str = unsafe_string(Ptr{UInt8}(sd))
    io = IOBuffer(sd_str)

    data = []
    i = 1
    while !eof(io)
        elem = MsgPack.unpack(io)
        if i % 2 == 1
            push!(data, Symbol(elem))
        else
            push!(data, elem)
        end
        i += 2
    end

    @assert length(data) % 2 == 0

    resultant_dict = Dict()
    i = 1
    while i <= length(data)
        key = data[i]
        value = data[i + 1]

        resultant_dict[key] = value
        i += 2
    end

    return resultant_dict
end
end

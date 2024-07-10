module OIFSerialization
export deserialize
import MsgPack

function deserialize(sd::Ptr{Cchar})::Dict
    sd_str = unsafe_string(sd)
    io = IOBuffer(sd_str)

    data = MsgPack.unpack(io)
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

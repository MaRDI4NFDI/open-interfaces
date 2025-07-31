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
        # Because MsgPack specification says that values _should_ be represented
        # in smallest number of bytes, Julia quite often unpacks
        # small integers in UInt8 values:
        # for example, original value `10` (of type `Int64`)
        # becomes `0x0a` (of type `UInt8`).
        # Because it breaks other things, we have to convert such values
        # deliberately to wider integer type.
        if typeof(elem) == UInt8
            elem = Int64(elem)
        end
        if i % 2 == 1
            push!(data, Symbol(elem))
        else
            push!(data, elem)
        end
        i += 1
    end

    @assert length(data) % 2 == 0

    resultant_dict = Dict()
    i = 1
    while i <= length(data)
        key = data[i]
        value = data[i+1]

        resultant_dict[key] = value
        i += 2
    end

    return resultant_dict
end
end

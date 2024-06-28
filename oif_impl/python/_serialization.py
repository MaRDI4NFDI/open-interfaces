from io import BytesIO

import msgpack


def deserialize(sd: bytes) -> dict:
    """Transform serialized dictionary `sd` to native Python dict."""

    bytes_stream = BytesIO(sd)
    unpacker = msgpack.Unpacker(bytes_stream, raw=False)

    data = []

    for datum in unpacker:
        data.append(datum)

    assert len(data) % 2 == 0, "Malformed serialized config dict"

    resultant_dict = {}
    i = 0
    while i < len(data):
        key = data[i]
        value = data[i + 1]

        resultant_dict[key] = value
        i += 2

    return resultant_dict

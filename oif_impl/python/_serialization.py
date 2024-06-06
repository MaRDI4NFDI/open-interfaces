from io import BytesIO

import msgpack
from oif.core import OIF_FLOAT64, OIF_INT, OIF_STR


def deserialize(sd: bytes) -> dict:
    """Transform serialized dictionary to native Python dict."""

    bytes_stream = BytesIO(sd)
    unpacker = msgpack.Unpacker(bytes_stream, raw=False)

    data = []

    for datum in unpacker:
        data.append(datum)

    assert len(data) % 3 == 0, "Malformed serialized config dict"

    resultant_dict = {}
    i = 0
    while i < len(data):
        key = data[i]
        dtype = data[i + 1]
        value = data[i + 2]

        assert dtype in [OIF_INT, OIF_FLOAT64, OIF_STR]

        resultant_dict[key] = value
        i += 3

    return resultant_dict

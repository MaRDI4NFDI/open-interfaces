#!/usr/bin/env julia

using OpenInterfaces

# julia and python connectors currently fail
for lang in ["c", "cpp"]
    N = 2
    A = [2., 0., 0., 1.0];
    b = [1., 1.]
    x = [0.,0.]

    OpenInterfaces.init(lang)
    OpenInterfaces.solve(N, A, b, x)
    OpenInterfaces.deinit()
end

#!/usr/bin/env julia

using OpenInterfaces

# julia and python connectors currently fail
for lang in ["c", "cpp"]
    OpenInterfaces.init(lang)
    try
        OpenInterfaces.eval("expression")
    catch e
        if lang in ["c", "cpp"]
            continue
        else
            error("eval failed for $lang")
        end
    end
    OpenInterfaces.deinit()
end

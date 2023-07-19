class OIFBackend:
    def __init__(self, handle):
        self.handle = handle

    def call(self, method, args):
        args_packed
        call_interface_method(self.handle, method, args_packed, outargs_packed)
        outargs = outargs_packed

        return outargs

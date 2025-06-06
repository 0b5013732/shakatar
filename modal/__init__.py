class _RemoteFunction:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

class App:
    def __init__(self, name: str | None = None):
        self.name = name
    def function(self):
        def decorator(fn):
            return _RemoteFunction(fn)
        return decorator
    def local_entrypoint(self):
        def decorator(fn):
            # Execute immediately when imported to mimic modal run
            fn()
            return fn
        return decorator

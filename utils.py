from functools import reduce, wraps


# Magic function to get torch model layers by name using dot notations
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


def add_dynamic_method(self):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        setattr(self, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator

"""
Miscellaneous utilities.
"""
from functools import wraps


def accepts(*types):
    """Decorator that performs argument validation for a function."""
    def check_accepts(f):
        # assert(len(types) == f.__code__.co_argcount)
        @wraps(f)
        def wrapper(*args, **kwargs):
            for (arg, t) in zip(args, types):
                if not isinstance(arg, t):
                    raise TypeError("arg:", arg, "is not of type:", t)
            return f(*args, **kwargs)
        return wrapper
    return check_accepts


def returns(ret_type):
    """Ensure the value returned by a function is of the correct type."""
    def check_return(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            if not isinstance(result, ret_type):
                raise TypeError("Invalid return type:", type(result), "expected:", ret_type)
            return result
        return wrapper
    return check_return


def convert(*types):
    """Convert function arguments via supplied types/conversion functions"""
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            new_args = [t(a) for t, a in zip(types, args)]
            return f(*new_args, **kwargs)
        return wrapper
    return decorate


def print_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        for i in args:
            print(i, type(i))
        for k, v in kwargs.items():
            print(k, v, type(v))
        return f(*args, **kwargs)
    return wrapper
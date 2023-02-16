from jinja2 import Environment, PackageLoader


def jinja_env():
    env = Environment(
        loader=PackageLoader("sklearn_evaluation", "assets/report"),
    )
    return env


class Range(object):
    """
    Set float range
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def in_range(self, n) -> bool:
        """
        Checks if n in range
        """
        return self.min <= n and self.max >= n


def run_if_args_are_not_none(func):
    """
    Runs a function only if given args are not none.
    Doesn't raise an error.
    """

    def wrapper(*args, **kw):
        if not any(elem is None for elem in args):
            return func(*args, **kw)

    return wrapper

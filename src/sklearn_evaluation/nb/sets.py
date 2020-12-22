"""
Operations in sets
"""


def differences(a, b):
    """Returns objects that are only on a, b and both
    """
    a = set(a)
    b = set(b)

    a_only = a - b
    b_only = b - a
    both = a & b

    return both, a_only, b_only

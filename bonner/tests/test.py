import numpy as np

from bonner.caching import cache


@cache(exclude_args=["x"], custom_identifier="test")
def add_1(x: int, y: str = "cat"):
    return np.zeros(x)


add_1(5)

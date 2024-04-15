"""
Testing the use of CPU vs GPU
"""

from numba import jit
from timeit import default_timer as timer


def func(n):
    """Normal function to run on cpu."""
    acc = []
    i = 0
    while i < n:
        acc.append(i)
        i += 1
    return acc


@jit(target_backend="cuda")
def func2(a):
    """Function to run on GPU using a decorator"""
    acc = []
    i = 0
    while i < n:
        acc.append(i)
        i += 1
    return acc


if __name__ == "__main__":
    n = 10000000

    # track cpu time
    start = timer()
    x = func(n)
    print(f'With Cpu: {timer() - start}')
    print(len(x))

    # track gpu time
    start = timer()
    y = func2(n)
    print(f'With Gpu: {timer() - start}')
    print(len(y))

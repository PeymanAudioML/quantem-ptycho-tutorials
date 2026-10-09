"""Minimal CuPy stand-in backed by NumPy so the authors' AmpflowS.py runs unmodified on CPU."""
import numpy as _np
from numpy import *  # noqa: F401,F403  (array, zeros, arange, matmul, abs, sum, nan_to_num, asarray, ...)

fft = _np.fft


def asnumpy(a):
    return _np.asarray(a)


class _Device:
    def __init__(self, n=0):
        self.n = n

    def use(self):
        return self


class cuda:  # noqa: N801
    Device = _Device


class _Pool:
    def free_all_blocks(self):
        pass

    def total_bytes(self):
        return 10 ** 12

    def used_bytes(self):
        return 0


def get_default_memory_pool():
    return _Pool()


def get_default_pinned_memory_pool():
    return _Pool()

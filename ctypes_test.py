import ctypes
import numpy.ctypeslib as ctl
from numpy.ctypeslib import ndpointer
import numpy as np

lib = ctypes.cdll.LoadLibrary("lib/sgbm_helper.so")
testsum = lib.test_sum
testsum.restype = None
testsum.argtypes = [ctl.ndpointer(np.uint8, flags = 'aligned, c_contiguous'),
                    ctl.ndpointer(np.uint64, flags = 'aligned, c_contiguous'),
                    ctypes.c_int,
                    ctypes.c_int]

in_vals = np.array(range(12), dtype = np.uint8).reshape(2,6)
out_vals = np.zeros((2,6), dtype = np.uint64)
testsum(in_vals, out_vals, in_vals.shape[0], in_vals.shape[1])
print(out_vals)




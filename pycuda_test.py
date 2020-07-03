import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule

build_options = ['-DD_STEP=2',
                 '--diag_warning=optimizations']
mod = SourceModule(open("lib/sgbm_helper.cu").read(), options=build_options)

multiply_them = mod.get_function("multiply_them")
three_d_matrix_test = mod.get_function("three_d_matrix_test")

#a = numpy.random.randn(400).astype(numpy.float32)
#b = numpy.random.randn(400).astype(numpy.float32)
N = 3
mat = np.array(range(N**3), dtype = np.float32).reshape((N,) * 3)
d_stride, r_stride, c_stride = mat.strides

three_d_matrix_test(
        drv.InOut(mat),
        np.int32(d_stride),
        np.int32(r_stride),
        np.int32(N),
        block = (N,N,N), grid=(1,1))

print(mat)
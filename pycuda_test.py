import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule(open("lib/sgbm_helper.cu").read())

multiply_them = mod.get_function("multiply_them")

#a = numpy.random.randn(400).astype(numpy.float32)
#b = numpy.random.randn(400).astype(numpy.float32)
N = 3
mat = np.array(range(N**3), dtype = np.float32).reshape((N,) * 3)
d_stride, r_stride, c_stride = mat.strides

three_d_matrix_test(
        drv.Out(mat),
        drv.In(d_stride),
        drv.In(r_stride),
        drv.In(c_stride), N)



#multiply_them(
#        drv.Out(dest), drv.In(a), drv.In(b),
#        block=(400,1,1), grid=(1,1))

print(dest)
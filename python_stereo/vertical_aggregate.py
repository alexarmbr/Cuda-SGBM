import unittest
from sgbm import SemiGlobalMatching, format_compiler_constants
import ctypes
import numpy.ctypeslib as ctl
from numpy.ctypeslib import ndpointer
import cv2
import matplotlib.pyplot as plt
from time import time
import os
import pdb
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import math
    
    
IMAGE_DIR = "Backpack-perfect"
im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
window_size=3, resize=(640,480))

params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
stereo.set_params(params)
stereo.params['ndisp'] = 50
# t1 = time()

# assert stereo.p1 is not None, "parameters have not been set"
# t1 = time()
# cim1 = stereo.census_transform(stereo.im1)
# cim2 = stereo.census_transform(stereo.im2)
# #print(f"census transform time {time() - t1}")

# if not stereo.reversed:
#     D = range(int(stereo.params['ndisp']))
# else:
#     D = reversed(range(int(-stereo.params['ndisp']), 1))
# cost_images = stereo.compute_disparity_img(cim1, cim2, D)
# cost_images = np.float32(cost_images)
##cost_images = cost_images[:,:,8:45]
#cost_images = cost_images[:,:,9:50]
shape = (480,640,51)
cost_images = np.float32(np.random.normal(loc=100, size=shape))
m, n, D = cost_images.shape
# direction == (1,0)
stereo.directions = [(1,0)]
L = stereo.aggregate_cost(cost_images)
L = L.transpose((2,0,1))

cost_images = cost_images.transpose((2,0,1))
cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)
d,rows,cols = cost_images.shape
d_step = 1

compiler_constants = {
    'D_STEP':d_step,
    'D':d,
    'ARR_SIZE':math.floor(d/d_step),
    'P1':5,
    'P2':90000
}
build_options = [format_compiler_constants(compiler_constants)] + ["--verbose"]
mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)


vertical_aggregate = mod.get_function("vertical_aggregate")
out = np.zeros_like(L)
out = np.ascontiguousarray(out, dtype = np.float32)
vertical_aggregate(drv.Out(out), drv.In(cost_images),
np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
drv.stop_profiler()
s1 = np.sum(np.float64(L))
s2 = np.sum(np.float64(out))



print("L sum: %f" % s1)
print("out sum: %f" % s2)
print(np.all(np.isclose(L, out)))
#pdb.set_trace()
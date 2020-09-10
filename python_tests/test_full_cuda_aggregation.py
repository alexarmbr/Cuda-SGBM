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
from time import time
IMAGE_DIR = "Adirondack-perfect"


if __name__ == "__main__":


    IMAGE_DIR = "Backpack-perfect"
    im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
    im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

    stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
    window_size=3, resize=(640,480))

    params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":False}
    stereo.set_params(params)
    stereo.params['ndisp'] = 50
    t1 = time()

    assert stereo.p1 is not None, "parameters have not been set"
    t1 = time()
    cim1 = stereo.census_transform(stereo.im1)
    cim2 = stereo.census_transform(stereo.im2)
    #print(f"census transform time {time() - t1}")
    
    if not stereo.reversed:
        D = range(int(stereo.params['ndisp']))
    else:
        D = reversed(range(int(-stereo.params['ndisp']), 1))
    cost_images = stereo.compute_disparity_img(cim1, cim2, D)
    m, n, D = cost_images.shape
    L = stereo.aggregate_cost(cost_images)
    #cost_images = np.float32(cost_images)
    
    #print("python aggregate cost %f" % (time() - t1))
    L = L.transpose((2,0,1))
    
    cost_images = cost_images.transpose((2,0,1))
    cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)
    d,rows,cols = cost_images.shape
    rows = np.int32(rows)
    cols = np.int32(cols)
    d = np.int32(d)


    d_step = 1
    
    compiler_constants = {
        'D_STEP':d_step,
        'D':d,
        'ARR_SIZE':math.floor(d/d_step),
        'P1':5,
        'P2':90000,
        'SHMEM_SIZE':64
    }
    build_options = [format_compiler_constants(compiler_constants)]
    mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)

    r_aggregate = mod.get_function('__r_aggregate')
    vertical_aggregate_down = mod.get_function('__vertical_aggregate_down')
    vertical_aggregate_up = mod.get_function('__vertical_aggregate_up')
    diagonal_br_tl_aggregate = mod.get_function('__diagonal_br_tl_aggregate')
    diagonal_tl_br_aggregate = mod.get_function('__diagonal_tl_br_aggregate')
    diagonal_tr_bl_aggregate = mod.get_function('__diagonal_tr_bl_aggregate')
    diagonal_bl_tr_aggregate = mod.get_function('__diagonal_bl_tr_aggregate')
    l_aggregate = mod.get_function('__l_aggregate')

    t1 = time()
    cost_images_ptr = drv.to_device(cost_images)
    dp_ptr = drv.mem_alloc(cost_images.nbytes)
    shmem_size = 16
    vertical_blocks = int(math.ceil(rows/shmem_size))
    #pdb.set_trace()

    vertical_aggregate_down(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))
    vertical_aggregate_up(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))
    
    r_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (shmem_size, shmem_size, 1), grid = (1, vertical_blocks))
    l_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (shmem_size, shmem_size, 1), grid = (1, vertical_blocks))
    
    diagonal_tl_br_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))
    diagonal_bl_tr_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))

    diagonal_tr_bl_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))
    diagonal_br_tl_aggregate(dp_ptr, cost_images_ptr, rows, cols, block = (256, 1, 1), grid = (1,1))
    
    
    
    
    agg_image = drv.from_device(dp_ptr, cost_images.shape, dtype = np.float32)
    print(f"aggregation time {time() - t1}")
    #agg_image = agg_image.transpose((2,1,0))
    min_cost_im = np.argmin(agg_image, axis=0)
    gt = np.argmin(L, axis=0)
    agg_image = agg_image.transpose((2,1,0))
    min_cost_im += 1
    gt += 1
    im = stereo.compute_depth(np.int32(min_cost_im))
    gt_im = stereo.compute_depth(np.int32(gt))
    ##im = stereo.normalize(im, 0.1)
    ##gt_im = stereo.normalize(gt_im, 0.1)
    np.save("../out_images/testim.npy", im)
    np.save("../out_images/gtim.npy", im)

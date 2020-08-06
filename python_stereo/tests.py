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





class TestCensusTransform(unittest.TestCase):
    """
    compare c++ implementation of census transform with
    python
    """
    def test_census_transform(self):
        
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))

        stereo = SemiGlobalMatching(im1, im1.copy(), os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(480,480))
        params = {"p1":5, "p2":30000, "census_kernel_size":7, "reversed":True}
        stereo.set_params(params)

        t1 = time()
        census1 = stereo.census_transform(stereo.im1.copy()) # python implementation
        #print(f"python census transform time: {time() - t1}")

        
        # 
        lib = ctypes.cdll.LoadLibrary("../lib/sgbm_helper.so")
        census = lib.census_transform
        census.restype = None
        census.argtypes = [ctl.ndpointer(np.uint8, flags = 'aligned, c_contiguous'),
                            ctl.ndpointer(np.uint64, flags = 'aligned, c_contiguous'),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int]
        census2 = np.zeros(stereo.im1.shape, dtype = np.uint64)
        t1 = time()
        census(stereo.im1.copy(),
        census2,
        stereo.im1.shape[0],
        stereo.im1.shape[1],
        3)
        #print(f"c++ census transform time: {time() - t1}")

        self.assertTrue(np.all(np.isclose(census1, census2)))

    def test_census_transform2(self):
        
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))

        stereo = SemiGlobalMatching(im1, im1.copy(), os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(480,480))
        params = {"p1":5, "p2":30000, "census_kernel_size":3, "reversed":True}
        stereo.set_params(params)

        
        testim = np.zeros((16,16), dtype = np.uint8)
        testim[7,7] = 5
        testim[7,8] = 9
        testim[7,6] = 3
        
        
        t1 = time()
        census1 = stereo.census_transform(testim.copy()) # python implementation
        #print(f"python census transform time: {time() - t1}")

        
        # 
        lib = ctypes.cdll.LoadLibrary("../lib/sgbm_helper.so")
        census = lib.census_transform
        census.restype = None
        census.argtypes = [ctl.ndpointer(np.uint8, flags = 'aligned, c_contiguous'),
                            ctl.ndpointer(np.uint64, flags = 'aligned, c_contiguous'),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int]
        census2 = np.zeros(testim.shape, dtype = np.uint64)
        t1 = time()
        census(testim.copy(),
        census2,
        testim.shape[0],
        testim.shape[1],
        1)
        #print(f"c++ census transform time: {time() - t1}")

        self.assertTrue(np.all(np.isclose(census1, census2)))
        


# TODO: horizontal is basically the same speed as vertical
# this probably means vertical can be sped up


class TestAggregation(unittest.TestCase):
        
    def test_horizontal_right(self):
        
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))
        #im1 = im1[:32,:,:]
        #im2 = im2[:32,:,:]

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
        stereo.set_params(params)
        stereo.params['ndisp'] = 50
        t1 = time()

        assert stereo.p1 is not None, "parameters have not been set"
        t1 = time()
        cim1 = stereo.census_transform(stereo.im1)
        cim2 = stereo.census_transform(stereo.im2)
        #print(f"census transform time {time() - t1}")
        #pdb.set_trace()
        if not stereo.reversed:
            D = range(int(stereo.params['ndisp']))
        else:
            D = reversed(range(int(-stereo.params['ndisp']), 1))
        cost_images = stereo.compute_disparity_img(cim1, cim2, D)
        cost_images = np.float32(cost_images)
        ##cost_images = cost_images[:,:,8:45]
        #cost_images = cost_images[:,:,9:50]
        #shape = (480,640,36)
        #cost_images = np.float32(np.random.normal(loc=100, size=shape))
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(0,1)]
        t1 = time()
        L = stereo.aggregate_cost_optimization_test(cost_images)
        print("python aggregate cost %f" % (time() - t1))
        L = L.transpose((2,0,1))
        
        cost_images = cost_images.transpose((2,0,1))
        cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)
        d,rows,cols = cost_images.shape
        d_step = 1
        
        shmem_size = 16
        compiler_constants = {
            'D_STEP':d_step,
            'D':d,
            'ARR_SIZE':math.floor(d/d_step),
            'P1':5,
            'P2':90000,
            'SHMEM_SIZE':shmem_size
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)



        r_aggregate = mod.get_function("r_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        vertical_blocks = int(math.ceil(rows/shmem_size))
        t1 = time()
        # pycuda complains when block size is greater than 32 x 32
        r_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (shmem_size,shmem_size,1), grid = (1,vertical_blocks)) 
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))

        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        t1 = [np.sum(out[:,:,i]) for i in range(out.shape[2])]
        t2 = [np.sum(out[:,i,:]) for i in range(out.shape[1])]
        t3 = [np.sum(out[i,:,:]) for i in range(out.shape[0])]
        t4 = [np.sum(L[:,:,i]) for i in range(L.shape[2])]
        t5 = [np.sum(L[:,i,:]) for i in range(L.shape[1])]
        t6 = [np.sum(L[i,:,:]) for i in range(L.shape[0])]
        diff1 = [t4[i] - a for i,a in enumerate(t1)]
        diff2 = [t5[i] - a for i,a in enumerate(t2)]
        diff3 = [t6[i] - a for i,a in enumerate(t3)]


        #pdb.set_trace()
        self.assertTrue(np.all(np.isclose(out, L)))

    def test_horizontal_left(self):
        
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))
        #im1 = im1[:32,:,:]
        #im2 = im2[:32,:,:]

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
        stereo.set_params(params)
        stereo.params['ndisp'] = 50
        t1 = time()

        assert stereo.p1 is not None, "parameters have not been set"
        t1 = time()
        cim1 = stereo.census_transform(stereo.im1)
        cim2 = stereo.census_transform(stereo.im2)
        #print(f"census transform time {time() - t1}")
        #pdb.set_trace()
        if not stereo.reversed:
            D = range(int(stereo.params['ndisp']))
        else:
            D = reversed(range(int(-stereo.params['ndisp']), 1))
        cost_images = stereo.compute_disparity_img(cim1, cim2, D)
        cost_images = np.float32(cost_images)
        ##cost_images = cost_images[:,:,8:45]
        #cost_images = cost_images[:,:,9:50]
        #shape = (480,640,36)
        #cost_images = np.float32(np.random.normal(loc=100, size=shape))
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(0,-1)]
        t1 = time()
        L = stereo.aggregate_cost_optimization_test(cost_images)
        print("python aggregate cost %f" % (time() - t1))
        L = L.transpose((2,0,1))
        
        cost_images = cost_images.transpose((2,0,1))
        cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)
        d,rows,cols = cost_images.shape
        d_step = 1
        
        shmem_size = 16
        compiler_constants = {
            'D_STEP':d_step,
            'D':d,
            'ARR_SIZE':math.floor(d/d_step),
            'P1':5,
            'P2':90000,
            'SHMEM_SIZE':shmem_size
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)



        l_aggregate = mod.get_function("l_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        vertical_blocks = int(math.ceil(rows/shmem_size))
        t1 = time()
        # pycuda complains when block size is greater than 32 x 32
        l_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (shmem_size,shmem_size,1), grid = (1,vertical_blocks)) 
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))

        print("L sum: %f" % s1)
        print("out sum: %f" % s2)


        #pdb.set_trace()
        self.assertTrue(np.all(np.isclose(out, L)))








    def test_vertical_down(self):
        
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        ##cost_images = cost_images[:,:,8:45]
        #cost_images = cost_images[:,:,9:50]
        #shape = (480,640,36)
        #cost_images = np.float32(np.random.normal(loc=100, size=shape))
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(1,0)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        vertical_aggregate = mod.get_function("vertical_aggregate_down")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        vertical_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))



        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        #pdb.set_trace()
        self.assertTrue(np.all(np.isclose(out, L)))


    def test_vertical_up(self):
        
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        ##cost_images = cost_images[:,:,8:45]
        #cost_images = cost_images[:,:,9:50]
        #shape = (480,640,36)
        #cost_images = np.float32(np.random.normal(loc=100, size=shape))
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(-1,0)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        vertical_aggregate = mod.get_function("vertical_aggregate_up")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        vertical_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))



        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        #pdb.set_trace()
        self.assertTrue(np.all(np.isclose(out, L)))




    def test_diagonal_tl_br(self):
            
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        ##cost_images = cost_images[:,:,8:45]
        #cost_images = cost_images[:,:,9:50]
        #shape = (480,640,36)
        #cost_images = np.float32(np.random.normal(loc=100, size=shape))
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(1,1)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        diagonal_aggregate = mod.get_function("diagonal_tl_br_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        diagonal_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))



        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        #pdb.set_trace()
        self.assertTrue(np.all(np.isclose(out, L)))



    def test_diagonal_tr_bl(self):
            
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(1,-1)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        diagonal_aggregate = mod.get_function("diagonal_tr_bl_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        diagonal_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))

        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        self.assertTrue(np.all(np.isclose(out, L)))




    def test_diagonal_br_tl(self):
            
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(-1,-1)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        diagonal_aggregate = mod.get_function("diagonal_br_tl_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        diagonal_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))

        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        self.assertTrue(np.all(np.isclose(out, L)))



    def test_diagonal_bl_tr(self):
            
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
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
        cost_images = np.float32(cost_images)
        m, n, D = cost_images.shape
        # direction == (1,0)
        stereo.directions = [(-1, 1)]
        t1 = time()
        L = stereo.aggregate_cost(cost_images)
        print("python aggregate cost %f" % (time() - t1))
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
            'P2':90000,
            'SHMEM_SIZE':64
        }
        build_options = [format_compiler_constants(compiler_constants)]
        mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
        
        
        diagonal_aggregate = mod.get_function("diagonal_bl_tr_aggregate")
        out = np.zeros_like(L)
        out = np.ascontiguousarray(out, dtype = np.float32)
        t1 = time()
        diagonal_aggregate(drv.Out(out), drv.In(cost_images),
        np.int32(rows), np.int32(cols), block = (256,1,1), grid = (1,1))
        print("cuda aggregate cost %f" % (time() - t1))
        drv.stop_profiler()
        s1 = np.sum(np.float64(L))
        s2 = np.sum(np.float64(out))

        print("L sum: %f" % s1)
        print("out sum: %f" % s2)
        self.assertTrue(np.all(np.isclose(out, L)))



# class Test3dMin(unittest.TestCase):

#     def test_slice_arr(self):
#         IMAGE_DIR = "Backpack-perfect"
#         im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
#         im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

#         stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
#         window_size=3, resize=(640,480))

#         params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
#         stereo.set_params(params)
#         stereo.params['ndisp'] = 50

#         cim1 = stereo.census_transform(stereo.im1)
#         cim2 = stereo.census_transform(stereo.im2)
#         if not stereo.reversed:
#             D = range(int(stereo.params['ndisp']))
#         else:
#             D = reversed(range(int(-stereo.params['ndisp']), 1))
#         cost_images = stereo.compute_disparity_img(cim1, cim2, D)
#         cost_images = cost_images.transpose((2,0,1))
#         cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)
        
#         #cost_images  = np.array(range(5*4*3), dtype = np.float32).reshape((5,4,3))
#         d, rows, cols = cost_images.shape
#         d_step = 1
#         build_options = []
#         mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
#         min_3d_mat = mod.get_function("slice_arr")
#         out = np.zeros((rows, cols), dtype=np.float32)
#         arr_slice = 44
#         min_3d_mat(drv.Out(out), drv.In(cost_images),
#         np.int32(rows), np.int32(cols), np.int32(arr_slice), block = (256,1,1), grid = (2,1))
#         self.assertTrue(np.all(np.isclose(out, cost_images[arr_slice,:,:])))



#     def test_min(self):
#         IMAGE_DIR = "Backpack-perfect"
#         im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
#         im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

#         stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
#         window_size=3, resize=(640,480))

#         params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
#         stereo.set_params(params)
#         stereo.params['ndisp'] = 50

#         cim1 = stereo.census_transform(stereo.im1)
#         cim2 = stereo.census_transform(stereo.im2)
#         if not stereo.reversed:
#             D = range(int(stereo.params['ndisp']))
#         else:
#             D = reversed(range(int(-stereo.params['ndisp']), 1))
#         cost_images = stereo.compute_disparity_img(cim1, cim2, D)
#         cost_images = cost_images.transpose((2,0,1))
#         cost_images = np.ascontiguousarray(cost_images, dtype = np.float32)

#         d, rows, cols = cost_images.shape
#         d_step = 2

#         compiler_constants = {
#             'D_STEP':2,
#             'D':d,
#             'ARR_SIZE':math.floor(d/d_step),
#             'P1':5,
#             'P2':90000
#         }
#         build_options = [format_compiler_constants(compiler_constants)]
#         mod = SourceModule(open("../lib/sgbm_helper.cu").read(), options=build_options)
#         min_3d_mat = mod.get_function("min_3d_mat")
#         out = np.zeros((rows, cols), dtype = np.float32)
#         min_3d_mat(drv.Out(out), drv.In(cost_images),
#         np.int32(rows), np.int32(cols), block = (256,1,1), grid = (2,1))
#         self.assertTrue(np.all(np.isclose(out, np.min(cost_images[::d_step, :, :], axis=0))))

        

        


if __name__ == "__main__":
    unittest.main()

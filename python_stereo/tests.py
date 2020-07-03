import unittest
from sgbm import SemiGlobalMatching
import ctypes
import numpy.ctypeslib as ctl
from numpy.ctypeslib import ndpointer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import os
import pdb
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

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
        print(f"python census transform time: {time() - t1}")

        
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
        print(f"c++ census transform time: {time() - t1}")

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
        print(f"python census transform time: {time() - t1}")

        
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
        print(f"c++ census transform time: {time() - t1}")

        self.assertTrue(np.all(np.isclose(census1, census2)))
        
        


class TestVerticalAggregation(unittest.TestCase):
    def testcase1(self):
        
        IMAGE_DIR = "Adirondack-perfect"
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
        print(f"census transform time {time() - t1}")
        
        if not stereo.reversed:
            D = range(int(stereo.params['ndisp']))
        else:
            D = reversed(range(int(-stereo.params['ndisp']), 1))
        cost_images = stereo.compute_disparity_img(cim1, cim2, D)
    
        L = np.zeros(cost_images.shape)
        m, n, D = cost_images.shape
        # direction == (1,0)
        u,v = (1,0)
        I = np.array([1] * n)
        J = np.array(range(n))

        while len(I) > 0:
            min_val = np.min(cost_images[I-u, J-v, :], axis = 1)
            for d in range(D):
                L[I,J,d] += cost_images[I, J, d] + stereo.dp_criteria(L[I-u, J-v, :], d, min_val)
            I+=u
            J+=v
            mask = np.logical_and(np.logical_and(0 <= I, I < m), np.logical_and(0 <= J, J < n)) # these are the paths that still have to traverse
            I = I[mask]
            J = J[mask]
        plt.imshow(np.argmin(L, axis=2))
        plt.show()

class Test3dMin(unittest.testcase):
    def test1(self):
        IMAGE_DIR = "Backpack-perfect"
        im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
        im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

        stereo = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
        window_size=3, resize=(640,480))

        params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
        stereo.set_params(params)
        stereo.params['ndisp'] = 50

        cim1 = stereo.census_transform(stereo.im1)
        cim2 = stereo.census_transform(stereo.im2)
        if not stereo.reversed:
            D = range(int(stereo.params['ndisp']))
        else:
            D = reversed(range(int(-stereo.params['ndisp']), 1))
        cost_images = stereo.compute_disparity_img(cim1, cim2, D)



        build_options = ['-DD_STEP=2',
                 '--diag_warning=optimizations']
        mod = SourceModule(open("lib/sgbm_helper.cu").read(), options=build_options)
        min_kernel = mod.get_function("min_3d_mat")
        rows, cols, d = stereo.im1.shape
        out = np.zeros((rows, cols))

        min_3d_mat(drv.Out(out), drv.In(stereo.im1),
        1, rows, cols, d)





        min_kernel(

        ) 
        


if __name__ == "__main__":
    unittest.main()

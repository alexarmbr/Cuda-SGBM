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


IMAGE_DIR = "Adirondack-perfect"

class TestHelperFunctions(unittest.TestCase):


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
        
        







if __name__ == "__main__":
    unittest.main()

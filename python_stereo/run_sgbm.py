from sgbm import SemiGlobalMatching
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import scipy
import time
#import pdb
import os



IMAGE_DIR = "Backpack-perfect"

if __name__ == "__main__":
    im1 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im1.png"))
    im2 = cv2.imread(os.path.join("../data", IMAGE_DIR ,"im0.png"))

    stereo4 = SemiGlobalMatching(im1, im2, os.path.join("../data", IMAGE_DIR ,"calib.txt"),
    window_size=3, resize=(640,480))


    params = {"p1":5, "p2":90000, "census_kernel_size":7, "reversed":True}
    stereo4.set_params(params)
    stereo4.params['ndisp'] = 50
    t1 = time.time()
    im = stereo4.compute_stereogram()
    out = stereo4.normalize(im, 0.1)
    print("sgbm time {:.2f}".format(time.time() - t1))
    plt.imshow(out)
    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


class _BasicStereo:

    def __init__(self, im1, im2, metadata_path,
    window_size = 11, stride_x = 1, stride_y = 1,
    plot_lines = False,resize = None):
        """
        im1, im2: two images taken by calibrated camera
        
        focal_length: focal length of camera used to take images,
        
        B: baseline, distance between camera when two images were taken. It is
        assumed that there is no vertical shift, i.e. camera was moved only in x direction
        
        metadata_path: path to file containing camera matrix, and ndisp value
        
        window_size: size of matching window, large window = more computation, should be odd
        
        stride_y,x: how many pixels to skip in between matching computation
        """
        #pdb.set_trace()
        self.im1 = im1
        self.im2 = im2
        if resize is not None:
            self.im1 = cv2.resize(self.im1, resize)
            self.im2 = cv2.resize(self.im2, resize)

        self.stride_x = stride_x
        self.stride_y = stride_y
        assert self.im1.shape == self.im2.shape, "image shapes must match exactly"
        assert window_size % 2 == 1, "window size should be odd number"
        self.window_size = window_size
        self.half_window_size = window_size // 2
        self.r, self.c = self.im1.shape[:2]
        self.params = self.parse_metadata(metadata_path)
        self.depth_im = np.zeros(self.r*self.c).reshape((self.r, self.c))
        self.plot_lines = plot_lines

        if self.plot_lines:
            self.j_indices = np.random.random_integers(0, self.c, 20)
            self.lines = []


    def parse_metadata(self, filename):
        d = {}
        with open(filename) as f:
            for line in f:
                (key, val) = line.split("=")
                val = val.strip("\n")
                try:
                    val = float(val)
                except:
                    pass
                d[key]=val
        d['focal_length'] = float(d['cam0'].strip("[").split(" ")[0])
        return d
    
    def pad_with_inf(self, img, direction, padding):
        """
        pad im to left or right with array of shape (im.shape[0], padding) of inf's
        """
        assert direction in {'left', 'right'}, "must pad to the left or right of image"
        pad = np.array([1e7] * (img.shape[0]*padding)).reshape(img.shape[0], padding)
        if direction == "left":
            img = np.hstack((pad,img))
        elif direction == "right":
            img = np.hstack((img, pad))
        return img

    
    def compute_stereogram(self):
        """
        wrapper around _compute_stereogram, in case you want to compute the stereogram both ways
        i.e. im1 -> im2 and im2 -> im1
        """
        self._compute_stereogram(self.im1, self.im2)


    def _compute_stereogram(self, im1, im2):
        """
        subclasses implement different stereo algorithms
        """
        assert False, "implement in subclass"


    
    def compute_depth(self, offset):
        """
        given offset of a point computes depth
        """
        return (self.params['focal_length'] * self.params['baseline']) / (offset + 0.01)


    def normalize(self, q):
        """
        replace all values less than quantile q with quantile q
        replace all values greater than quantile 1-1 with quantile 1-q

        """
        lower, upper = tuple(np.quantile(self.depth_im, (q, 1-q)))
        self.depth_im[self.depth_im < lower] = lower
        self.depth_im[self.depth_im > upper] = upper

    
    def mse(self, cutout_a, cutout_b):
        """
        compute mse between two cutouts
        """
        diff = np.float32(cutout_a) - np.float32(cutout_b)
        diff **= 2
        return np.mean(diff)
    
    def save_stereogram(self, imname):
        np.save(imname, self.depth_im)
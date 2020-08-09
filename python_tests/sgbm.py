from BaseStereo import _BasicStereo
import pdb
import cv2
import numpy as np
import time as t

def format_compiler_constants(d):
    """
    d - dictionary of key value pairs containing variable name a and val
    """
    #return "-D"+ ", ".join([f'{k}={v}' for k,v in d.items()])
    return "-D"+ ", ".join(['%s=%d' % (k,v) for k,v in d.items()])



class SemiGlobalMatching(_BasicStereo):

    def __init__(self, *args, **kwargs):
        """
        Semi Global Matching stereo algorithm with hamming distance
        https://core.ac.uk/download/pdf/11134866.pdf

        Arguments:
            census_kernel_size {int} -- kernel size used to create census image
            
        """

        super().__init__(*args, **kwargs)

        self.im1 = cv2.cvtColor(self.im1, cv2.COLOR_BGR2GRAY)
        self.im2 = cv2.cvtColor(self.im2, cv2.COLOR_BGR2GRAY)
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        self.census_images = {}
        
        self.census_kernel_size = None
        self.csize = None
        self.p1 = None
        self.p2 = None
        self.reversed = None

    def set_params(self, param_dict):
        """
        sets parameters

        Arguments:
            param_dict {dictionary} -- dictionary containing required parameters
        """
        if 'p1' in param_dict:
            self.p1 = param_dict['p1']
        if 'p2' in param_dict:
            self.p2 = param_dict['p2']
        if 'census_kernel_size' in param_dict:
            self.census_kernel_size = param_dict['census_kernel_size']
            self.csize = self.census_kernel_size // 2
            assert self.census_kernel_size % 2 == 1\
                and self.census_kernel_size < 8,\
                    "census kernel size needs to odd and less than 8"
        if 'reversed' in param_dict:
            self.reversed = param_dict['reversed']



    def compute_stereogram(self):
        self.set_params({'reversed':False})
        stereo1 = self._compute_stereogram(self.im1, self.im2)
        #self.set_params({'reversed':True})
        #stereo2 = self._compute_stereogram(self.im2, self.im1)
        #stereo1 = self.normalize(stereo1, 0.1)
        #stereo2 = self.normalize(stereo2, 0.1)
        #stereo = (stereo1 + stereo2) / 2
        return stereo1

    
    def compute_disparity_img(self, cim1, cim2, disparity_range):
        """
        compute pixelwise hamming distance between 2 census images

        Arguments:
            cim1 {np.ndarray} -- census image 1
            cim2 {np.ndarray} -- census image 2
            disparity_range {generator} -- range of ints to compute disparity over
        """
        cost_images = []
        t1 = t.time()
        for d in disparity_range:
            if d != 0:
                if d < 0:
                    shifted_im2 = cim2[:, :d].copy() # cut off right
                    shifted_im1 = cim1[:, -d:].copy() # cut off left
                else:
                    shifted_im2 = cim2[:, d:].copy() # cut off left
                    shifted_im1 = cim1[:, :-d].copy() # cut off right
            else:
                shifted_im1 = cim1.copy()
                shifted_im2 = cim2.copy()
            cost_im = self.hamming_distance(shifted_im1, shifted_im2)
            #cost_im = np.abs(shifted_im1 - shifted_im2)

            if d > 0:
                cost_im = self.pad_with_inf(cost_im, "right", d)
            elif d < 0:
                cost_im = self.pad_with_inf(cost_im, "left", -d)
    
            cost_images.append(cost_im)
        cost_images = np.stack(cost_images)
        cost_images = cost_images.transpose(1,2,0)
        #print(f"shift and stack time: {t.time() - t1}")
        return cost_images

    


    
    def _compute_stereogram(self, im1, im2):
        """
        compute disparity image that warps im2 -> im1

        Arguments:
            im1 {np.ndarray} -- image 1
            im2 {np.ndarray} -- image 2
        """
        assert self.p1 is not None, "parameters have not been set"
        t1 = t.time()
        cim1 = self.census_transform(im1)
        cim2 = self.census_transform(im2)
        #print(f"census transform time {t.time() - t1}")
        
        if not self.reversed:
            D = range(int(self.params['ndisp']))
        else:
            D = reversed(range(int(-self.params['ndisp']), 1))
        cost_images = self.compute_disparity_img(cim1, cim2, D)
        
        t1 = t.time()
        cost_images = self.aggregate_cost(cost_images)
        #print(f"aggregate cost time: {t.time() - t1}")
        t1 = t.time()
        min_cost_im = np.argmin(cost_images, axis=2)
        #print(f"argmin time: {t.time() - t1}")
        min_cost_im += 1
        #min_cost_im = cv2.medianBlur(np.float32(min_cost_im), 3)
        min_cost_im = np.int32(min_cost_im)
        return self.compute_depth(min_cost_im)


    def normalize(self, img, q):
        """
        replace all values less than quantile q with quantile q
        replace all values greater than quantile 1-1 with quantile 1-q

        """
        lower, upper = tuple(np.quantile(img, (q, 1-q)))
        img[img < lower] = lower
        img[img > upper] = upper
        return img

    def census_transform(self, image, imname = None):
        """
        compute census image using kernel of size csize

        Arguments:
            image {np.ndarray} -- greyscale image to compute census image of
            imname {string} -- name for image to save census image as
        """
        census_image = np.zeros(image.shape)

        for i in range(self.csize, image.shape[0] - self.csize):
            for j in range(self.csize, image.shape[1] - self.csize):
                cutout = image[i-self.csize:i+self.csize+1, j-self.csize:j+self.csize+1]
                mid = cutout[self.csize, self.csize]
                census = cutout >= mid # which surrounding pixels are greater than this pixel
                # this mask is transformed to a binary number which is used as a signature for this pixel
                 
                census = census.reshape(1,-1).squeeze()
                mid = len(census) // 2
                census = np.delete(census, mid) # remove middle element
                census_val = np.uint64(0)

                #if (i > 100 and j > 100):
                #    pdb.set_trace()

                one = np.uint64(1)
                zero = np.uint64(0)
                for B in census:
                    census_val <<= one
                    census_val |= one if B else zero
                census_image[i,j] = census_val
        census_image = np.uint64(census_image)
        return census_image


    def hamming_distance(self, cim1, cim2):
        """
        Compute elementwise hamming distance between two census images,
        each pixel in the image is treated as a binary number

        Arguments:
            cim1 {np.ndarray} -- census image 1
            cim2 {np.ndarray} -- census image 2
        """
        assert cim1.shape == cim2.shape, "inputs must have same shape"
        z = np.zeros(cim1.shape)
        xor = np.bitwise_xor(cim1, cim2)
        
        while not (xor == 0).all():
            z+=xor & 1
            xor = xor >> 1
        return z

    def aggregate_cost_optimization_test(self, cost_array):
        """
        aggregate cost over 8 paths using dp algorithm
        try precomputing all the mins over d in advance
        this lends itself to much more efficient cuda implementation
        and does not hurt quality of image very much (barely affects it)

        Arguments:
            cost_array {np.ndarray} -- array of shape (h,w,d) that contains pixel wise costs (hamming distances) for each d
        """
        L = np.zeros(cost_array.shape, dtype=np.float32)
        m, n, D = cost_array.shape
        for (u,v) in self.directions:
            I,J = self.get_starting_indices((u,v), (m,n))
            count = 0
            while len(I) > 0:
                if count % 16 == 0:
                    cum_min = np.min(L, axis = 2)
                min_val = cum_min[I-u, J-v]
                for d in range(D):
                    L[I,J,d] += cost_array[I, J, d] + self.dp_criteria(L[I-u, J-v, :], d, min_val)
                I+=u
                J+=v
                mask = np.logical_and(np.logical_and(0 <= I, I < m), np.logical_and(0 <= J, J < n)) # these are the paths that still have to traverse
                I = I[mask]
                J = J[mask]
                count += 1
        return L

    def aggregate_cost(self, cost_array):
        """
        aggregate cost over 8 paths using dp algorithm

        Arguments:
            cost_array {np.ndarray} -- array of shape (h,w,d) that contains pixel wise costs (hamming distances) for each d
        """
        L = np.zeros(cost_array.shape, dtype=np.float32)
        m, n, D = cost_array.shape
        for (u,v) in self.directions:
            I,J = self.get_starting_indices((u,v), (m,n))
            while len(I) > 0:
                min_val = np.min(L[I-u, J-v, :], axis = 1)
                for d in range(D):
                    L[I,J,d] += cost_array[I, J, d] + self.dp_criteria(L[I-u, J-v, :], d, min_val)
                I+=u
                J+=v
                mask = np.logical_and(np.logical_and(0 <= I, I < m), np.logical_and(0 <= J, J < n)) # these are the paths that still have to traverse
                I = I[mask]
                J = J[mask]
        return L
        
        


    def get_starting_indices(self, direction, im_shape):
        """
        generates starting array indices for cost aggregation using sweep direction
        and shape of cost surface

        Arguments:
            dir {tuple} -- direction of aggregation along cost surface
            im_shape {tuple} -- 2-d shape of cost surface, not including disparity dimension
        """
        m,n = im_shape
        i_direction, j_direction = direction
        assert (all([abs(i) < 2 for i in direction])), "Invalid Direction!"

        if direction == (1,0):
            I = np.array([1] * n)
            J = np.array(range(n))
        elif direction == (-1, 0):
            I = np.array([m-2] * n)
            J = np.array(range(n))
        elif direction == (0,1):
            I = np.array(range(m))
            J = np.array([1] * m)
        elif direction == (0,-1):
            I = np.array(range(m))
            J = np.array([n-2] * m)
        elif direction == (1,1):
            I = np.concatenate((np.array(range(1,m)), np.array([1] * (n-2))))
            J = np.concatenate((np.array([1] * (m-1)), np.array(range(2,n))))
        elif direction == (1,-1):
            I = np.concatenate((np.array(range(1,m)), np.array([1] * (n-1))))
            J = np.concatenate((np.array([n-2] * (m-1)), np.array(range(n-1))))
        elif direction == (-1, 1):
            I = np.concatenate((np.array(range(m-1)), np.array([m-2] * (n-2))))
            J = np.concatenate((np.array([1] * (m-1)), np.array(range(2, n))))
        elif direction == (-1, -1):
            I = np.concatenate((np.array(range(m-1)), np.array([m-2] * (n-2))))
            J = np.concatenate((np.array([n-2] * (m-1)), np.array(range(n-2))))
        return I,J

    
    def dp_criteria(self, disparity_costs, d, prev_min):
        """
        generates cost associated with neighboring cell according to 
        criteria explained in paper

        Arguments:
            disparity_costs {np.ndarray} -- costs of each disparity from all adjacent cells
            d {int} -- current disparity to compute
            prev_min {float} -- minimum cost of disparity from adjacent cell to scale current cell by
        """

        d1 = disparity_costs[:, d]
        if d-1 >= 0:
            d2 = disparity_costs[:, d-1] + self.p1
        else:
            d2 = np.array([float("inf")] * disparity_costs.shape[0])
        
        if d+1 < disparity_costs.shape[1]:
            d3 = disparity_costs[:, d+1] + self.p1
        else:
            d3 = np.array([float("inf")] * disparity_costs.shape[0])
        d4 = prev_min + self.p2
        costs = np.vstack((d1, d2, d3, d4)).T
        return np.min(costs, axis=1) - prev_min
        

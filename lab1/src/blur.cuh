/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef BLUR_DEVICE_CUH
#define BLUR_DEVICE_CUH

#include "cuda_header.cuh"

// This function will conduct the convolution for a particular thread index
// given all the other inputs. (See how this function is called in blur.cu
// to get an understanding of what it should do.
CUDA_CALLABLE void cuda_blur_kernel_convolution(int thread_index,
                                                float* gpu_raw_data,
                                                float* gpu_blur_v,
                                                float* gpu_out_data,
                                                int n_frames,
                                                int blur_v_size);

// This function will be called from the host code to invoke the kernel
// function. Any memory address/pointer locations passed to this function
// must be host addresses. This function will be in charge of allocating
// GPU memory, invoking the kernel, and cleaning up afterwards. The result
// will be stored in out_data. The function returns the amount of time that
// it took for the function to complete (prior to returning) in milliseconds.
float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size);

#endif

/**
 * CUDA-implemented utility functions & kernels needed by the neural net
 * @author Aadyot Bhatngar
 * @date April 22, 2018
 */

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <algorithm>
#include "helper_cuda.h"

// CUDA block width
#define BW 1024

/**
 * Sets all entries in a device buffer of floats equal to a specified value.
 */
template<typename T> void cudaMemsetType(T *dev_ptr, T val, int n_vals)
{
    thrust::device_ptr<T> thrust_dev_ptr(dev_ptr);
    thrust::fill(thrust_dev_ptr, thrust_dev_ptr + n_vals, val);
}

/**
 * Invokes a CUDA kernel to compute the average cross entropy between softmaxed
 * predictions pred_Y and ground truth true_Y.
 *
 * @param pred_Y predictions made by model (probability vectors)
 * @param true_Y true output values (one-hot vectors)
 * @param n number of predictions
 * @param c number of channels per prediction
 * @param h height of each prediction
 * @param w width of each prediction
 *
 * @return cross-entropy loss between pred_Y and true_Y
 */
float CrossEntropyLoss(float* pred_Y, float* true_Y, int n, int c, int h, int w)
{
    // Inialize loss on the device to be zero
    float loss, *d_loss;
    CUDA_CALL( cudaMalloc(&d_loss, sizeof(float)) );
    cudaMemsetType<float>(d_loss, 0.0, 1);

    // Accumulate the total loss on the device by invoking a kernel
    int n_blocks = std::min(65535, (n * c * h * w + BW  - 1) / BW);
    // TODO (set 5): call CrossEntropyKernel

    // Copy back the accumulated loss on the device back to the host
    CUDA_CALL( cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaFree(d_loss) );

    // Return the average loss
    return loss;
}

/**
* Invokes a CUDA kernel to compute the average accuracy of softmaxed predictions
* pred_Y, given ground truth true_Y.
*
* @param pred_Y predictions made by model (probability vectors)
* @param true_Y true output values (one-hot vectors)
* @param n number of predictions
* @param c number of channels per prediction
* @param h height of each prediction
* @param w width of each prediction
*
* @return proportion of n for which the maximum entry in pred_Y (most probable
*         class predicted) is the same as the one entry in true_Y (true class)
*/
float SoftThresholdAccuracy(float* pred_Y, float* true_Y,
    int n, int c, int h, int w)
{
    // Initialize the accuracy on the device to be zero
    float acc, *d_acc;
    CUDA_CALL( cudaMalloc(&d_acc, sizeof(float)) );
    cudaMemsetType<float>(d_acc, 0.0, 1);

    // Accumulate the total loss on the device by invoking a kernel
    int n_blocks = std::min(65535, (n * c * h * w + BW - 1) / BW);
    SoftThresholdAccKernel<<<n_blocks, BW, BW * sizeof(float)>>>(pred_Y, true_Y,
        d_acc, n, c, h, w);

    // Copy back the accumulated accuracy on the device back to the host
    CUDA_CALL(cudaMemcpy(&acc, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_acc));

    // Return the average accuracy
    return acc / static_cast<float>(n);
}



/**
 * Kernel to compute cross-entropy between pred_Y and true_Y as described by
 * {\link CrossEntropyLoss}.
 */
__global__ void CrossEntropyKernel(float* pred_Y, float* true_Y, float *loss,
    int n, int c, int h, int w)
{
    extern __shared__ float shmem[];

    // TODO (set 5): use a parallel reduction to compute cross-entropy between
    //               pred_Y and true_Y, i.e. -sum( log(pred_Y[i]) * true_Y[i] ),
    //               where i ranges from 0 to (n*c*h*w) - 1

    // atomically add the accumulated loss per block into the global accumulator
    if (threadIdx.x == 0)
        atomicAdd(loss, shmem[0] / static_cast<float>(n));
}

/**
 * Kernel to compute accuracy of pred_Y given ground truth true_Y as described
 * by {\link SoftThresholdAccuracy}.
 */
__global__ void SoftThresholdAccKernel(float* pred_Y, float* true_Y, float* acc,
    int n, int c, int h, int w)
{
    extern __shared__ float shmem[];
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;

    // have each thread in each block accumulate some of the total loss in
    // shared memory
    shmem[tid] = 0.0;
    for (; idx < n; idx += blockDim.x * gridDim.x)
    {
        unsigned idx_cur = idx * c * h * w;

        // Determine which copmonent/element of the current prediction vector
        // and its corresponding ground truth is largest
        unsigned argmax_pred = 0, argmax_true = 0;
        for (unsigned j = 0; j < c * h * w; ++j)
        {
            if (pred_Y[idx_cur + argmax_pred] < pred_Y[idx_cur + j])
                argmax_pred = j;

            if (true_Y[idx_cur + argmax_true] < true_Y[idx_cur + j])
                argmax_true = j;
        }

        // If we were correct, add 1 to the accuracy count
        if (argmax_pred == argmax_true)
            shmem[tid] += 1.0;
    }
    __syncthreads();

    // do a reduction to sum up all of the accuracy components in this block's
    // shared memory
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shmem[tid] += shmem[tid + s];
        __syncthreads();
    }

    // atomically add the accumulated accuracy per block into the global accumulator
    if (tid == 0) atomicAdd(acc, shmem[tid]);
}

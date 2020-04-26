/**
 * Header file for CUDA-implemented utilities needed by the neural net
 * @author Aadyot Bhatngar
 * @date April 22, 2018
 */

#pragma once

#include <cuda_runtime.h>

template<typename T> void cudaMemsetType(T *dev_ptr, T val, int n_vals);

float CrossEntropyLoss(float *pred_Y, float *true_Y,
    int n, int c, int h, int w);

float SoftThresholdAccuracy(float *pred_Y, float *true_Y,
    int n, int c, int h, int w);

__global__ void CrossEntropyKernel(float *pred_Y, float *true_Y, float *loss,
    int n, int c, int h, int w);

__global__ void SoftThresholdAccKernel(float *pred_Y, float *true_Y, float *acc,
    int n, int c, int h, int w);

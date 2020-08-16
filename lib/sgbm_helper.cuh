

#ifndef SGBM_DEVICE
#define SGBM_DEVICE

#include "cuda_runtime.h"

__device__ float dp_criteria(float *dp, int ind, int depth_dim_size, int d, float P_one, float P_two, float * d_zero, float * d_one, float * d_two, float * d_three);


__global__ void r_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void l_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void vertical_aggregate_down(float *dp, float *cost_image, int m, int n);

__global__ void vertical_aggregate_up(float *dp, float *cost_image, int m, int n);

__global__ void diagonal_tl_br_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void diagonal_tr_bl_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void diagonal_br_tl_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void diagonal_bl_tr_aggregate(float *dp, float *cost_image, int m, int n);


#endif
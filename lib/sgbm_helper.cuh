

#ifndef SGBM_DEVICE
#define SGBM_DEVICE

#include <cuda_runtime.h>

#define D_STEP 1
#define D 101
#define ARR_SIZE 51
#define P1 5
#define P2 90000
#define SHMEM_SIZE 32


__device__ float dp_criteria(float *dp, int ind, int depth_dim_size, int d, float P_one, float P_two, float * d_zero, float * d_one, float * d_two, float * d_three);

__global__ void __shift_subtract_stack(unsigned int * L,
    unsigned int * R,
    float * out,
    int rows, int cols);

__global__ void __shift_subtract_stack_2(unsigned int * L,
    unsigned int * R,
    float * out,
    int rows, int cols);


__global__ void __shift_subtract_stack_3(unsigned int * L,
    unsigned int * R,
    float * out,
    int rows, int cols);


__global__ void __r_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __l_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __vertical_aggregate_down(float *dp, float *cost_image, int m, int n);

__global__ void __vertical_aggregate_up(float *dp, float *cost_image, int m, int n);

__global__ void __diagonal_tl_br_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __diagonal_tr_bl_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __diagonal_br_tl_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __diagonal_bl_tr_aggregate(float *dp, float *cost_image, int m, int n);

__global__ void __argmin_3d_mat(float * dp, int * stereo_im, int m, int n);




// wrapper funcs
float * device_shift_subtract_stack(unsigned int * L, unsigned int * R,
    float * out,
    int rows, int cols);
float * r_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * l_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * vertical_aggregate_down(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * vertical_aggregate_up(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * diagonal_tl_br_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * diagonal_tr_bl_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * diagonal_br_tl_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
float * diagonal_bl_tr_aggregate(int nCols, int nRows, float * shifted_images, float * dp, cudaStream_t stream);
int * argmin(int nCols, int nRows, float * dp, int * stereo_im, cudaStream_t stream);



#endif
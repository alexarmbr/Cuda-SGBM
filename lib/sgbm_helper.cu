//#define DEPTH 2 
#include <limits>

__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

__global__ void three_d_matrix_test(float *dest, int stride_depth, int stride_row, int N)
{
  if(threadIdx.x < my_global_var){
  //if((threadIdx.x == threadIdx.y) & (threadIdx.x == threadIdx.z)){
    //int idx = (threadIdx.z * stride_depth * stride_row) + (threadIdx.y * stride_row) + threadIdx.x;
    int idx = (threadIdx.z * my_global_var * my_global_var) + (threadIdx.y * my_global_var) + threadIdx.x;
    dest[idx] = 100.0f;
  //}
}
}

// dp - cost aggregation array
// cost_image - m x n x D array
// d - use every d channels of input to conserve register memory
// m - image rows
// n - image columns
// D - depth
// depth_stride - pitch along depth dimension
// row_stride - pitch along row dimension

// takes min along depth dimension, puts output in dp
__global__ void min_3d_mat(float *dp, float *cost_image, 
int d,
int m, int n, int D)
{
  col = blockDim.x * blockIdx.x + threadIdx.x
  
  K = ceil(n / (blockDim.x * gridDim.x));
  D_SIZE = floor(D / D_STEP); 
  float arr[D_SIZE];

  while(col < m)
  {
    for (int row = 1; row < n; row++)
    {
      arr_ind = 0
      
      #pragma unroll
      for (int depth = 0; depth < D; depth+=D_STEP){
        arr[arr_ind] = cost_image[depth * m * n + row * n + col];
        arr_ind++;
      }
      dp[depth * m * n + row * n + col] == arr_min(&arr);
    }
  col += blockDim.x
  }

}



__device__ float arr_min(float * arr)
{
  float a = std::numeric_limits<float>::infinity():

  for(int i = 0; i < D_SIZE; i++)
    a = fminf(arr[i], a);
  return a;
}


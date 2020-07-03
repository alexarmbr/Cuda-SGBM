#define MYVAR 2 


__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

__global__ void three_d_matrix_test(float *dest, int stride_depth, int stride_row, int N)
{
  if(threadIdx.x < N){
  //if((threadIdx.x == threadIdx.y) & (threadIdx.x == threadIdx.z)){
    //int idx = (threadIdx.z * stride_depth * stride_row) + (threadIdx.y * stride_row) + threadIdx.x;
    int idx = (threadIdx.z * N * N) + (threadIdx.y * N) + threadIdx.x;
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
// __global__ void vertical_aggregate(float *dp, float *cost_image, 
// int d,
// int m, int n, int D)
// {
//   K = ceil(n / (blockDim.x * gridDim.x))

//   for (int k = 0; k < K; k += blockDim.x)
//   {

//     float arr[d];
//     #pragma unroll
//     for (int i = 0; i < D; i+=d){
//       arr[d]
  
//     }
//   }










}



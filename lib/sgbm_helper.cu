//#define DEPTH 2
#include <stdio.h>
// __global__ void multiply_them(float *dest, float *a, float *b)
// {
//   const int i = threadIdx.x;
//   int arr[10];
//   //arr[i] = 20;
//   dest[i] = a[i] * b[i] + arr[i];
// }

// __global__ void three_d_matrix_test(float *dest, int stride_depth, int stride_row, int N)
// {
//   if(threadIdx.x < my_global_var){
//   //if((threadIdx.x == threadIdx.y) & (threadIdx.x == threadIdx.z)){
//     //int idx = (threadIdx.z * stride_depth * stride_row) + (threadIdx.y * stride_row) + threadIdx.x;
//     int idx = (threadIdx.z * my_global_var * my_global_var) + (threadIdx.y * my_global_var) + threadIdx.x;
//     dest[idx] = 100.0f;

// }
// }

// dp - cost aggregation array
// cost_image - m x n x D array
// d - use every d channels of input to conserve register memory
// m - image rows
// n - image columns
// D - depth
// depth_stride - pitch along depth dimension
// row_stride - pitch along row dimension


__device__ float dp_criteria(float *dp, int ind, int depth_dim_size, int d, float P_one, float P_two, float * d_zero, float * d_one, float * d_two, float * d_three){
     *d_zero = dp[ind];
     if (d > 0)
       *d_one = dp[ind - depth_dim_size] + P_one;
     else
       *d_one = 10000000;
     
     if (d < D-1)
       *d_two = dp[ind + depth_dim_size] + P_one;
     else
       *d_two = 10000000;
     return fminf(fminf(*d_zero, *d_one), fminf(*d_two, *d_three)) - *d_three + P_two;

  }

// takes min along depth dimension, puts output in dp
// __global__ void min_3d_mat(float * dp, float *cost_image, 
// int m, int n)
// {
//   int col = blockDim.x * blockIdx.x + threadIdx.x;
  
//   int K = ceilf(n / (blockDim.x * gridDim.x));
//   int D_SIZE = floorf(D / D_STEP); 
//   float arr[ARR_SIZE];

//   while(col < n)
//   {
//     for (int row = 0; row < m; row++)
//     {
//       int arr_ind = 0;
      
//       //#pragma unroll
//       for (int depth = 0; depth < D; depth+=D_STEP){
//         arr[arr_ind] = cost_image[depth * m * n + row * n + col];
//         arr_ind++;
//       }
//       dp[row * n + col] = arr_min(arr, D_SIZE);
//     }
//   col += blockDim.x;
//   }

// }


// __global__ void slice_arr(float *dp, float *cost_image, 
//   int m, int n, int slice)
//   {
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
  
//     while(col < n)
//     {
//       for (int row = 0; row < m; row++)
//       {
//         dp[row * n + col] = cost_image[slice * m * n + row * n + col];
//       }
//     col += blockDim.x;
//     }
  
//   }


__global__ void r_aggregate_naive(float *dp, float *cost_image, int m, int n)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int depth_dim_size = m*n;
  while(row < m)
  {

    for (int col=1; col < n; col++)
    {
        float prev_min = 100000000.0;
        int ind = row * n + col - 1;
        
        for (int depth = 0; depth < D; depth+=D_STEP){
          prev_min = fminf(cost_image[ind], prev_min);
          ind += (depth_dim_size * D_STEP);
        }

        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = prev_min + (float) P2;
        ind = row * n + col - 1;
        int current_ind = row * n + col;

        // todo: try having this loop go from 1 to d-1 and removing the if else
        for (int d = 0; d < D; d+=D_STEP){
          // for each d I need dp[{d-1, d, d+1}, row-1, col],
          dp[current_ind] = cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
          ind += (depth_dim_size * D_STEP);
          current_ind += (depth_dim_size * D_STEP);
        }
    }
    row += blockDim.x;
  }
}

__global__ void r_aggregate(float *dp, float *cost_image, int m, int n)
{
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x;
  int depth_dim_size = m*n;
  __shared__ float MinArray[SHMEM_SIZE][SHMEM_SIZE];
  int K = 0; // this variable keeps track of the progress in aggregating
  // across the columns of the image

  while ((col < n) & (row < m))
  {
    int ind = row * n + col;
    int prev_min = 100000000.0;
  
    for (int depth = 0; depth < D; depth+=D_STEP){
      prev_min = fminf(dp[ind], prev_min);
      ind += (depth_dim_size * D_STEP);
    }
  
    MinArray[threadIdx.y][threadIdx.x] = prev_min;
    __syncthreads();
  
  
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;

    // threads from only one warp will handle rightward aggregation across the
    // region that has been loaded into shared memory
    // for threads where threadIdx.y is 0, now threadIdx.x will index the rows
    if (threadIdx.y == 0)
    {
      int agg_row = threadIdx.x + blockIdx.y * blockDim.y;
      int start_K = K;
      int local_K = 0;
      
      if (agg_row < m)
      {
        for(; (K < (n - 1)) && (K < (start_K + SHMEM_SIZE)); K++)
        {
          //printf("K: %d, local_K: %d \n", K, local_K);
          float d3 = MinArray[threadIdx.x][local_K] + (float) P2;
          
          int ind = agg_row * n + K;
          int next_ind = agg_row * n + K + 1;

          for (int d = 0; d < D; d+=D_STEP){
            dp[next_ind] = cost_image[next_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            ind += (depth_dim_size * D_STEP);
          }
          local_K++;
        }
      }

    }

    // this sync could probably be removed, only has effect for thread with lowest threadIdx.x in each block
    //__syncthreads();
    if (threadIdx.y == 0 && threadIdx.x == 31)
    printf("col: %d, K: %d \n", col, K);
    col+=blockDim.x;
    
    


  }

}








  
__global__ void vertical_aggregate(float *dp, float *cost_image, 
  int m, int n)
  {
    // which column of array to work on
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int depth_dim_size = m*n;
    
    // todo: maybe it will work better to take running average of every d 
    // slices
    while(col < n)
    {
      for (int row = 1; row < m; row++)
      {
        //int arr_ind = 0;
        float prev_min = 100000000.0;
        int ind = (row - 1) * n + col;
        
        // calculate min cost disparity for this column from row-1
        //#pragma unroll
        for (int depth = 0; depth < D; depth+=D_STEP){
          prev_min = fminf(dp[ind], prev_min);
          ind += (depth_dim_size * D_STEP);
          //arr[arr_ind] = cost_image[depth * m * n + (row - 1) * n + col];
          //arr_ind++;
        }

        //  float prev_min = arr_min(arr, D_SIZE);
        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = prev_min + (float) P2;
        ind = (row - 1) * n + col;
        int current_ind = row * n + col;

        
        // todo: try having this loop go from 1 to d-1 and removing the if else
        for (int d = 0; d < D; d+=D_STEP){
          // for each d I need dp[{d-1, d, d+1}, row-1, col],
          dp[current_ind] = cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
          ind += (depth_dim_size * D_STEP);
          current_ind += (depth_dim_size * D_STEP);
        }
      }
    col += blockDim.x;
    }
  }













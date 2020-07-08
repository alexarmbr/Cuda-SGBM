//#define DEPTH 2

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
__device__ float arr_min(float * arr, int dsize)
{
  float a = 10000000;

  for(int i = 0; i < dsize; i++)
    a = fminf(arr[i], a);
  return a;
}

// takes min along depth dimension, puts output in dp
__global__ void min_3d_mat(float *dp, float *cost_image, 
int m, int n)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  
  int K = ceilf(n / (blockDim.x * gridDim.x));
  int D_SIZE = floorf(D / D_STEP); 
  float arr[ARR_SIZE];

  while(col < n)
  {
    for (int row = 0; row < m; row++)
    {
      int arr_ind = 0;
      
      //#pragma unroll
      for (int depth = 0; depth < D; depth+=D_STEP){
        arr[arr_ind] = cost_image[depth * m * n + row * n + col];
        arr_ind++;
      }
      dp[row * n + col] = arr_min(arr, D_SIZE);
    }
  col += blockDim.x;
  }

}


__global__ void slice_arr(float *dp, float *cost_image, 
  int m, int n, int slice)
  {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    //int K = ceilf(n / (blockDim.x * gridDim.x));
    //int D_SIZE = floorf(D / D_STEP); 
    //float arr[ARR_SIZE];
  
    while(col < n)
    {
      for (int row = 0; row < m; row++)
      {
        dp[row * n + col] = cost_image[slice * m * n + row * n + col];
      }
    col += blockDim.x;
    }
  
  }


  
__global__ void vertical_aggregate(float *dp, float *cost_image, 
  int m, int n)
  {
    // which column of array to work on
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    // max number of columns a thread will work on
    int K = ceilf(n / (blockDim.x * gridDim.x));
    
    // how large local array of disparity values will be
    // how many disparity values will be aggregated over
    int D_SIZE = floorf(D / D_STEP);
    //float arr[ARR_SIZE];
    
    // todo: maybe it will work better to take running average of every d 
    // slices
    while(col < n)
    {
      for (int row = 1; row < m; row++)
      {
        //int arr_ind = 0;
        float prev_min = 100000000.0;
        
        // calculate min cost disparity for this column from row-1
        //#pragma unroll
        for (int depth = 0; depth < D; depth+=D_STEP){
          prev_min = fminf(cost_image[depth * m * n + (row - 1) * n + col], prev_min);
          //arr[arr_ind] = cost_image[depth * m * n + (row - 1) * n + col];
          //arr_ind++;
        }

        //  float prev_min = arr_min(arr, D_SIZE);
        float d0 = 0;
        float d1 = 0;
        float d2 = 0;
        float d3 = prev_min + (float) P2;
        
        // todo: try having this loop go from 1 to d-1 and removing the if else
        for (int d = 0; d < D; d+=D_STEP){
          // for each d I need dp[{d-1, d, d+1}, row-1, col], 
          d0 = dp[d * m * n + (row - 1) * n + col];
          if (d > 0)
            d1 = dp[(d-1) * m * n + (row - 1) * n + col] + (float) P1;
          else
            d1 = 10000000;
          
          if (d < D-1)
          d2 = dp[(d+1) * m * n + (row - 1) * n + col] + (float) P1;
          else
            d2 = 10000000;
          dp[d * m * n + row * n + col] = cost_image[d * m * n + row * n + col] + fminf(fminf(d0,d1), fminf(d2,d3)) - prev_min;
        }
      }
    col += blockDim.x;
    }
  }












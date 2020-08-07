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
          prev_min = fminf(dp[ind], prev_min);
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



// TODO, what if m % SHMEM_SIZE != 0 ?
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
    float prev_min = 100000000.0;
    
    for (int depth = 0; depth < D; depth+=D_STEP){
      prev_min = fminf(dp[ind], prev_min);
      ind += (depth_dim_size * D_STEP);
    }
  
    MinArray[threadIdx.y][threadIdx.x] = prev_min;
    __syncthreads();
  
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    
    // when processing a video stream, need to make sure that processing of multiple
    // frames can overlap, since after this point only one warp of threads is executing

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
          float d3 = MinArray[threadIdx.x][local_K] + (float) P2;

          int ind = agg_row * n + K + 1;
          for (int d = 0; d < D; d+=D_STEP){
            dp[ind] += cost_image[ind] + dp_criteria(dp, ind-1, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            //dp[ind] = cost_image[ind] + dp[ind - 1];
            ind += (depth_dim_size * D_STEP);
          }
          local_K++;
        }
      }
    }

    __syncthreads();
    col+=blockDim.x;

  }

}

__global__ void l_aggregate(float *dp, float *cost_image, int m, int n)
{
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = n - 1 - threadIdx.x;
  int depth_dim_size = m*n;
  __shared__ float MinArray[SHMEM_SIZE][SHMEM_SIZE];
  int K = n-1; // this variable keeps track of the progress in aggregating
  // across the columns of the image

  while ((col >= 0) & (row < m))
  {
    int ind = row * n + col;
    float prev_min = 100000000.0;
    
    for (int depth = 0; depth < D; depth+=D_STEP){
      prev_min = fminf(dp[ind], prev_min);
      ind += (depth_dim_size * D_STEP);
    }
  
    MinArray[threadIdx.y][SHMEM_SIZE - 1 - threadIdx.x] = prev_min;
    __syncthreads();
  
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    
    // when processing a video stream, need to make sure that processing of multiple
    // frames can overlap, since after this point only one warp of threads is executing

    // threads from only one warp will handle rightward aggregation across the
    // region that has been loaded into shared memory
    // for threads where threadIdx.y is 0, now threadIdx.x will index the rows
    if (threadIdx.y == 0)
    {
      int agg_row = threadIdx.x + blockIdx.y * blockDim.y;
      int start_K = K;
      int local_K = SHMEM_SIZE - 1;

      if (agg_row < m)
      {
        for(; (K > 0) && (K > (start_K - SHMEM_SIZE)); K--)
        {
          float d3 = MinArray[threadIdx.x][local_K] + (float) P2;

          int ind = agg_row * n + K - 1;
          for (int d = 0; d < D; d+=D_STEP){
            dp[ind] += cost_image[ind] + dp_criteria(dp, ind+1, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            //dp[ind] = cost_image[ind] + dp[ind - 1];
            ind += (depth_dim_size * D_STEP);
          }
          local_K--;
        }
      }
    }

    __syncthreads();
    col-=blockDim.x;
    
  }

}


  
__global__ void vertical_aggregate_down(float *dp, float *cost_image, 
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
          dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
          ind += (depth_dim_size * D_STEP);
          current_ind += (depth_dim_size * D_STEP);
        }
      }
    col += blockDim.x;
    }
  }


  __global__ void vertical_aggregate_up(float *dp, float *cost_image, 
    int m, int n)
    {
      // which column of array to work on
      int col = blockDim.x * blockIdx.x + threadIdx.x;
      int depth_dim_size = m*n;
      
      // todo: maybe it will work better to take running average of every d 
      // slices
      while(col < n)
      {
        for (int row = m-2; row >= 0; row--)
        {
          //int arr_ind = 0;
          float prev_min = 100000000.0;
          int ind = (row + 1) * n + col;
          
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
          ind = (row + 1) * n + col;
          int current_ind = row * n + col;
  
          
          // todo: try having this loop go from 1 to d-1 and removing the if else
          for (int d = 0; d < D; d+=D_STEP){
            // for each d I need dp[{d-1, d, d+1}, row-1, col],
            dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            ind += (depth_dim_size * D_STEP);
            current_ind += (depth_dim_size * D_STEP);
          }
        }
      col += blockDim.x;
      }
    }  


  // aggregation along diagonal from top left to bottom right
  __global__ void diagonal_tl_br_aggregate(float *dp, float *cost_image, 
    int m, int n)
    {
      // which column of array to work on
      int start_col = blockDim.x * blockIdx.x + threadIdx.x + 1;
      int depth_dim_size = m*n;
      
      // todo: maybe it will work better to take running average of every d 
      // slices
      while(start_col < n)
      {
        int col = start_col;
        for (int row = 1; row < m; row++)
        {
          //int arr_ind = 0;
          float prev_min = 100000000.0;
          int ind = (row - 1) * n + col - 1;
          
          // calculate min cost disparity for this column from row-1
          //#pragma unroll
          for (int depth = 0; depth < D; depth+=D_STEP){
            prev_min = fminf(dp[ind], prev_min);
            ind += (depth_dim_size * D_STEP);
          }
  
          float d0 = 0;
          float d1 = 0;
          float d2 = 0;
          float d3 = prev_min + (float) P2;
          ind = (row - 1) * n + col - 1;
          int current_ind = row * n + col;
  
          
          // todo: try having this loop go from 1 to d-1 and removing the if else
          for (int d = 0; d < D; d+=D_STEP){
            // for each d I need dp[{d-1, d, d+1}, row-1, col],
            dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            ind += (depth_dim_size * D_STEP);
            current_ind += (depth_dim_size * D_STEP);
          }

          col += 1;
          if (col == n) // wrap each thread around once it gets to the last column
            col = 1;

        }
      start_col += blockDim.x;
      }
    }

// aggregation along diagonal from top right to bottom left
    __global__ void diagonal_tr_bl_aggregate(float *dp, float *cost_image, 
      int m, int n)
      {
        // which column of array to work on
        // thread with blockIdx.x == 0 and threadIdx.x == 0 will start at column n-2 (and aggregate
        // using data from columns n-1)
        int start_col = (n - 2) - (blockDim.x * blockIdx.x) - threadIdx.x;
        int depth_dim_size = m*n;
        
        // todo: maybe it will work better to take running average of every d 
        // slices
        while(start_col >= 0)
        {
          int col = start_col;
          for (int row = 1; row < m; row++)
          {
            //int arr_ind = 0;
            float prev_min = 100000000.0;
            int ind = (row - 1) * n + col + 1;
            
            // calculate min cost disparity for this column from row-1
            //#pragma unroll
            for (int depth = 0; depth < D; depth+=D_STEP){
              prev_min = fminf(dp[ind], prev_min);
              ind += (depth_dim_size * D_STEP);
            }
    
            float d0 = 0;
            float d1 = 0;
            float d2 = 0;
            float d3 = prev_min + (float) P2;
            ind = (row - 1) * n + col + 1;
            int current_ind = row * n + col;
    
            
            // todo: try having this loop go from 1 to d-1 and removing the if else
            for (int d = 0; d < D; d+=D_STEP){
              // for each d I need dp[{d-1, d, d+1}, row-1, col],
              dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
              ind += (depth_dim_size * D_STEP);
              current_ind += (depth_dim_size * D_STEP);
            }
  
            col -= 1;
            if (col < 0) // wrap each thread around once it gets to the last column
              col = n-2;
  
          }
        start_col -= blockDim.x;
        }
      }



// aggregation along diagonal from bottom right to top left
    __global__ void diagonal_br_tl_aggregate(float *dp, float *cost_image, 
      int m, int n)
      {
        // which column of array to work on
        // thread with blockIdx.x == 0 and threadIdx.x == 0 will start at column n-2 (and aggregate
        // using data from columns n-1)
        int start_col = (n - 2) - (blockDim.x * blockIdx.x) - threadIdx.x;
        int depth_dim_size = m*n;
        
        // todo: maybe it will work better to take running average of every d 
        // slices
        while(start_col >= 0)
        {
          int col = start_col;
          for (int row = m-2; row >= 0; row--)
          {
            //int arr_ind = 0;
            float prev_min = 100000000.0;
            int ind = (row + 1) * n + col + 1;
            
            // calculate min cost disparity for this column from row-1
            //#pragma unroll
            for (int depth = 0; depth < D; depth+=D_STEP){
              prev_min = fminf(dp[ind], prev_min);
              ind += (depth_dim_size * D_STEP);
            }

            float d0 = 0;
            float d1 = 0;
            float d2 = 0;
            float d3 = prev_min + (float) P2;
            ind = (row + 1) * n + col + 1;
            int current_ind = row * n + col;

            
            // todo: try having this loop go from 1 to d-1 and removing the if else
            for (int d = 0; d < D; d+=D_STEP){
              // for each d I need dp[{d-1, d, d+1}, row-1, col],
              dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
              ind += (depth_dim_size * D_STEP);
              current_ind += (depth_dim_size * D_STEP);
            }

            col -= 1;
            if (col < 0) // wrap each thread around once it gets to the last column
              col = n-2;

          }
        start_col -= blockDim.x;
        }
      }




  // aggregation along diagonal from top left to bottom right
  __global__ void diagonal_bl_tr_aggregate(float *dp, float *cost_image, 
    int m, int n)
    {
      // which column of array to work on
      int start_col = blockDim.x * blockIdx.x + threadIdx.x + 1;
      int depth_dim_size = m*n;
      
      // todo: maybe it will work better to take running average of every d 
      // slices
      while(start_col < n)
      {
        int col = start_col;
        for (int row = m-2; row >= 0; row--)
        {
          //int arr_ind = 0;
          float prev_min = 100000000.0;
          int ind = (row + 1) * n + col - 1;
          
          // calculate min cost disparity for this column from row-1
          //#pragma unroll
          for (int depth = 0; depth < D; depth+=D_STEP){
            prev_min = fminf(dp[ind], prev_min);
            ind += (depth_dim_size * D_STEP);
          }
  
          float d0 = 0;
          float d1 = 0;
          float d2 = 0;
          float d3 = prev_min + (float) P2;
          ind = (row + 1) * n + col - 1;
          int current_ind = row * n + col;
  
          
          // todo: try having this loop go from 1 to d-1 and removing the if else
          for (int d = 0; d < D; d+=D_STEP){
            // for each d I need dp[{d-1, d, d+1}, row-1, col],
            dp[current_ind] += cost_image[current_ind] + dp_criteria(dp, ind, depth_dim_size, d, (float) P1, (float) P2, &d0, &d1, &d2, &d3);
            ind += (depth_dim_size * D_STEP);
            current_ind += (depth_dim_size * D_STEP);
          }

          col += 1;
          if (col == n) // wrap each thread around once it gets to the last column
            col = 1;

        }
      start_col += blockDim.x;
      }
    }





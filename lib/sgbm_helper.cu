__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

__global__ void three_d_matrix_test(float *dest, int stride_depth, int stride_row, int N)
{
  if(threadIdx.x < N){
  if((threadIdx.x == threadIdx.y) & (threadIdx.x == threadIdx.z)){
    idx = threadIdx.z * stride_depth * stride_row + threadIdx.y * stride_row + threadIdx.x
    dest[idx] = 100.0f
  }
}
}


//__global__ void vertical_aggregate(float *L, float *cost_image, 
//int u, int v,
//int m, int n, int D)



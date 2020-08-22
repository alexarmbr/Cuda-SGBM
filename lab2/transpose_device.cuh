#ifndef CUDA_TRANSPOSE_CUH
#define CUDA_TRANSPOSE_CUH

enum TransposeImplementation { NAIVE, SHMEM, OPTIMAL };

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type);

#endif

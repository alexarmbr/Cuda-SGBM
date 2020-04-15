/* 
 * CUDA cuda_header
 * Loko Kung, 2018
 */

#ifndef CUDA_HEADER_CUH_
#define CUDA_HEADER_CUH_

#ifdef __CUDA_ARCH__

// Device function attributes
#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__

#else

// Host function attributes
#define CUDA_CALLABLE

#endif // __CUDA_ARCH__

#endif // CUDA_HEADER_CUH_

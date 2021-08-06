#ifndef SGBM_HELPER
#define SGBM_HELPER

#include "sgbm_helper.cuh"
#include <iostream>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



extern "C" void _sgbm(cv::Mat * im1,
cv::Mat * im2,
int * depth_im,
int nRows,
int nCols,
cudaStream_t stream = 0);



extern "C" void test_sum(const uint8_t * in_arr,
unsigned long long int * out_arr,
int rows, int cols);


extern "C" void census_transform(const uint8_t * in_arr,
unsigned long long int * out_arr,
int rows, int cols, int csize);


inline float hamming_dist(unsigned long long a,
unsigned long long b);


extern "C" void shift_subtract_stack(unsigned long long int * L,
unsigned long long int * R,
float * out,
int rows, int cols);


extern "C" void census_transform_mat(Mat * in_arr,
unsigned long long int * out_arr,
int rows, int cols, int csize);


#endif
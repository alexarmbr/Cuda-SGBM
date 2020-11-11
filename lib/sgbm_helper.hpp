#include <iostream>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;




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
int rows, int cols, int D);


extern "C" void census_transform_mat(Mat * in_arr,
unsigned long long int * out_arr,
int rows, int cols, int csize);
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lib/sgbm_helper.hpp"
//#include "cuda_runtime.h"
#include "lib/sgbm_helper.cuh"
#include <iostream>
#include <chrono>

#define D 51

// pkg-config --cflags opencv4 --libs


using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout <<" Usage: sgbm Image1 Image2" << endl;
     return -1;
    }

    Mat im1;
    Mat im2;
    im1 = imread(argv[1], IMREAD_GRAYSCALE);
    im2 = imread(argv[2], IMREAD_GRAYSCALE);
    resize(im1, im1, Size(480,480));
    resize(im2, im2, Size(480,480));

    if(! im1.data | ! im2.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    CV_Assert(im1.depth() == CV_8U);
    int nRows = im1.rows;
    int nCols = im1.cols;

    //if (im1.isContinuous())
    //{
    //    nCols *= nRows;
    //    nRows = 1;
    //}

    unsigned long long int * cim1;
    unsigned long long int * cim2;
    float * shifted_images;
    cim1 = new unsigned long long int [nCols * nRows];
    cim2 = new unsigned long long int [nCols * nRows];
    shifted_images = new float [nCols * nRows * D];
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    census_transform_mat(&im1, cim1, nRows, nCols, 7);
    census_transform_mat(&im2, cim2, nRows, nCols, 7);
    shift_subtract_stack(cim1, cim2, shifted_images, nRows, nCols, D);
    

    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // TODO: which is this much slower that previous implementation
    //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // cudaMalloc modifies this pointer to point to block of memory on device
    float* gpu_ptr_shifted_im;
    cudaMalloc((void **) &gpu_ptr_shifted_im, sizeof(float) * nCols * nRows * D);
    cudaMemcpy(gpu_ptr_shifted_im, shifted_images, sizeof(float) * nCols * nRows * D, cudaMemcpyHostToDevice);
      
    float* gpu_ptr_agg_im;
    cudaMalloc((void **) &gpu_ptr_agg_im, sizeof(float) * nCols * nRows * D);
    cudaMemset(gpu_ptr_agg_im, 0, sizeof(float) * nCols * nRows * D);

    gpu_ptr_agg_im = r_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = l_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = vertical_aggregate_down(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = vertical_aggregate_up(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_tl_br_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_tr_bl_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_br_tl_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_bl_tr_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);

    return 0;
}
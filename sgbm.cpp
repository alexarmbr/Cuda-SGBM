#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lib/sgbm_helper.hpp"

#include "lib/sgbm_helper.cuh"
#include <iostream>
#include <chrono>

// pkg-config --cflags opencv4 --libs

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



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


    unsigned long long int * cim1;
    unsigned long long int * cim2;
    float * shifted_images;
    cim1 = new unsigned long long int [nCols * nRows];
    cim2 = new unsigned long long int [nCols * nRows];
    shifted_images = new float [nCols * nRows * D];
    
    // census transform both images
    census_transform_mat(&im1, cim1, nRows, nCols, 3);
    census_transform_mat(&im2, cim2, nRows, nCols, 3);

    // if shifted_images, cim1, cim2 were numpy arrays with dimensions disparity (D), row, col
    // shifted_images[D,i,j] = cim1[i,j+D] - cim2[i,j]
    // with padding where necessary
    shift_subtract_stack(cim1, cim2, shifted_images, nRows, nCols, D);
    
    // cudaMalloc modifies this pointer to point to block of memory on device
    float* gpu_ptr_shifted_im;
    gpuErrchk( cudaMalloc((void **) &gpu_ptr_shifted_im, sizeof(float) * nCols * nRows * D) );
    gpuErrchk( cudaMemcpy(gpu_ptr_shifted_im, shifted_images, sizeof(float) * nCols * nRows * D, cudaMemcpyHostToDevice) );
      
    // aggregated image will go here
    float* gpu_ptr_agg_im;
    gpuErrchk( cudaMalloc((void **) &gpu_ptr_agg_im, sizeof(float) * nCols * nRows * D) );
    gpuErrchk( cudaMemset(gpu_ptr_agg_im, 0, sizeof(float) * nCols * nRows * D) );

    // each direction of aggregation
    gpu_ptr_agg_im = r_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = l_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = vertical_aggregate_down(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = vertical_aggregate_up(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_tl_br_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_tr_bl_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_br_tl_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpu_ptr_agg_im = diagonal_bl_tr_aggregate(nCols, nRows, gpu_ptr_shifted_im, gpu_ptr_agg_im);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );



    // argmin
    int * stereo_im;
    gpuErrchk( cudaMalloc((void **) &stereo_im, sizeof(int) * nCols * nRows) );
    gpuErrchk( cudaMemset(stereo_im, 0, sizeof(int) * nCols * nRows) );
    argmin(nCols, nRows, gpu_ptr_agg_im, stereo_im);
    int * stereo_im_host = new int[nCols * nRows];
    cudaMemcpy(stereo_im_host, stereo_im, sizeof(int) * nCols * nRows, cudaMemcpyDeviceToHost);


    FILE *f = fopen("stereo_im.data", "wb");
    fwrite(stereo_im_host, sizeof(int), nCols * nRows, f);
    fclose(f);

    return 0;
}
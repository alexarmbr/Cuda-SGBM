
#include <iostream>
#include <opencv2/core/core.hpp>
#include "sgbm_helper.hpp"

using namespace std;
using namespace cv;




extern "C" void _sgbm(cv::Mat * im1,
cv::Mat * im2,
int * depth_im,
int nRows,
int nCols,
int stream)
{
    unsigned long long int * cim1;
    unsigned long long int * cim2;
    float * shifted_images;
    cim1 = new unsigned long long int [nCols * nRows];
    cim2 = new unsigned long long int [nCols * nRows];
    shifted_images = new float [nCols * nRows * D];
    
    // census transform both images
    census_transform_mat(im1, cim1, nRows, nCols, 3);
    census_transform_mat(im2, cim2, nRows, nCols, 3);

    // if shifted_images, cim1, cim2 were numpy arrays with dimensions disparity (D), row, col
    // shifted_images[D,i,j] = cim1[i,j+D] - cim2[i,j]
    // with padding where necessary
    shift_subtract_stack(cim1, cim2, shifted_images, nRows, nCols);
    
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
    //int * stereo_im_host = new int[nCols * nRows];
    cudaMemcpy(depth_im, stereo_im, sizeof(int) * nCols * nRows, cudaMemcpyDeviceToHost);


}






extern "C" void test_sum(const uint8_t * in_arr,
unsigned long long int * out_arr,
int rows, int cols)
{
    //unsigned long long val = 0
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            out_arr[i * cols + j] |= (unsigned long long) (in_arr[i * cols + j] >= 6);   
        }
    }
}



extern "C" void census_transform(const uint8_t * in_arr,
unsigned long long int * out_arr,
int rows, int cols, int csize)
{
    // loop over image
    for(int i = csize; i < rows-csize; i++){
        for(int j = csize; j < cols-csize; j++){
            uint8_t mid = in_arr[i * cols + j];
            unsigned long long census_val = 0;
            
            // loop over census region
            for(int u = -1 * csize; u <= csize; u++){
                for(int v = -1 * csize; v<=csize; v++){

                    if (!((u==0) && (v==0)))
                    {
                        census_val <<= (unsigned long long) 1;
                        census_val |= (unsigned long long) (in_arr[((i+u) * cols) + j + v] >= mid);
                    }
                }
            }
            out_arr[i * cols + j] = census_val;
        }
    }
}

inline float hamming_dist(unsigned long long a,
unsigned long long b){
    unsigned long long c = a^b;
    float z = 0;
    while (c != 0){
        z += c & 1;
        c>>=1;
    }
    return z;
}



extern "C" void shift_subtract_stack(unsigned long long int * L,
unsigned long long int * R,
float * out,
int rows, int cols){

    int d = -1;
    int imsize = rows * cols;
    for(int i = 0; i < imsize * D; i++){
        
        // compute for next disparity
        if (i % (rows * cols) == 0)
            d+=1;
        
        // no corresponding pixels at this disparity
        if (i % cols > cols - 1 - d)
            out[i] = 1e7;
        
        // hamming distance between right image shifted by d and left image
        else
            out[i] = hamming_dist(R[(i % imsize)+d], L[i % imsize]);
    }
}





extern "C" void census_transform_mat(Mat * in_arr,
unsigned long long int * out_arr,
int rows, int cols, int csize)
{
    // loop over image
    for(int i = csize; i < rows-csize; i++){
        for(int j = csize; j < cols-csize; j++){
            unsigned char mid = (unsigned char) (*in_arr).at<uint8_t>(i,j);
            unsigned long long census_val = 0;
            //std::cout << i << "\n";
            // loop over census region
            unsigned long long tmp;
            for(int u = -1 * csize; u <= csize; u++){
                for(int v = -1 * csize; v<=csize; v++){

                    if (!((u==0) && (v==0)))
                    {
                        census_val <<= (unsigned long long) 1;
                        tmp = (unsigned long long) ((*in_arr).at<uint8_t>(i+u,j+v) >= mid);
                        census_val |= tmp;
                        //census_val |= (unsigned long long) ((*in_arr).data[((i+u) * cols) + j + v] >= mid);
                    }
                }
            }
            out_arr[i * cols + j] = census_val;
        }
    }
}
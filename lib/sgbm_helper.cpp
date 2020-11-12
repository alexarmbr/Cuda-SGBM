
#include <iostream>
#include <opencv2/core/core.hpp>
#include "sgbm_helper.hpp"

using namespace std;
using namespace cv;

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
int rows, int cols, int D){

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
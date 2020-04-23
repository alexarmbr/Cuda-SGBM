#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <fstream>
#include "fft_convolve.cuh"

using namespace std;
int main(int argc, char ** argv){
    
    cufftComplex arr[22];
    for (int i=0; i < 22; i++){
        arr[i].x = i;
    }

    cufftComplex * data;
    cudaMalloc((void **) &data, sizeof(cufftComplex) * 22);
    cudaMemcpy(data, arr, 22 * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    float * maxval;
    *maxval = 983;
    cudaCallMaximumKernel(1,3,data,maxval,22);

    cufftComplex result_arr[22];
    cudaMemcpy(result_arr, data, 22 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 22; i++){
        cout << i <<"th value is: " << result_arr[i].x << endl;
    }

}
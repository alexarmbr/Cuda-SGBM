#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cufft.h>
#include <fstream>
#include "fft_convolve.cuh"

using namespace std;



int main(int argc, char ** argv){
    

    int size = 49;
    cufftComplex arr[size];
    for (int i=0; i < size; i++){
        arr[i].x = i;
    }


    cufftComplex * data;
    cudaMalloc((void **) &data, sizeof(cufftComplex) * size);
    cudaMemcpy(data, arr, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);


    float *dev_max_abs_val;
    cudaMalloc((void **) &dev_max_abs_val, sizeof(float));
    cudaMemset(dev_max_abs_val, 0, sizeof(float));

    cudaCallMaximumKernel(2,12,data,dev_max_abs_val,size);

    float maxval;
    cudaMemcpy(&maxval, dev_max_abs_val, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "maximum value is: " << maxval << endl;
    
    cudaError err = cudaGetLastError();
    if  (cudaSuccess != err){
            cerr << "Error " << cudaGetErrorString(err) << endl;
    } else {
            cerr << "No kernel error detected" << endl;
    }
    // int f = 5;
    // int * five = &f;
    // int out = myfunc(five);
    // cout << "return val: " << out;

}



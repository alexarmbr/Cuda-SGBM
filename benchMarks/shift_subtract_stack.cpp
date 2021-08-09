
#include <opencv2/core/core.hpp>
#include "sgbm_helper.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <limits>

const int NUM_ITERS = 1;
const int NUM_STREAMS = 1;

// unsigned long long * genData(int size)
// {

//     unsigned long long * arr = new unsigned long long [size];
//     std::random_device rd;     
//     std::mt19937_64 eng(rd());
//     std::uniform_int_distribution<unsigned long long> distr;

//     for(int i = 0; i < size; i++)
//         arr[i] = distr(eng);
//     return arr;
// }


unsigned int * genData(int size)
{
    unsigned int * arr = new unsigned int [size];
    for(int i = 0; i < size; i++)
        arr[i] = (unsigned int) rand();
    return arr;
}




int main( int argc, char** argv )
{

    int rows = 2048;
    int cols = 1024;
    unsigned int * L[NUM_STREAMS];
    unsigned int * R[NUM_STREAMS];
    float * shifted_images[NUM_STREAMS];

    for(int i = 0; i < NUM_STREAMS; i++)
    {
        L[i] = genData(rows * cols);
        R[i] = genData(rows * cols);
        shifted_images[i] = new float [rows * cols * D];
    }
    

    // cpu benchmark
    // auto clock_start = std::chrono::system_clock::now();
    // for(int iter = 0; iter < 1; iter++)
    // {
    //     std::cout << "cpu iter: " << iter << std::endl;
    //     shift_subtract_stack(L[0], R[0], shifted_images[0], rows, cols);
    // }
    // auto clock_end = std::chrono::system_clock::now();
    // unsigned int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end-clock_start).count();
    // float avg_time = ((float) elapsed_time / (float) NUM_ITERS);
    // std::cout << "average cpu time: " << avg_time << std::endl;

    unsigned int * im1_gpu;
    unsigned int * im2_gpu;
    float * shifted_images_gpu;
    
    auto clock_start_gpu = std::chrono::system_clock::now();
    for(int iter = 0; iter < NUM_ITERS; iter++)
    {
        std::cout << "gpu iter: " << iter << std::endl;
        gpuErrchk( cudaMalloc((void **) &shifted_images_gpu, sizeof(float) * rows * cols * D) );
        gpuErrchk( cudaMalloc((void **) &im1_gpu, sizeof(unsigned int) * rows * cols) );
        gpuErrchk( cudaMalloc((void **) &im2_gpu, sizeof(unsigned int) * rows * cols) );
        gpuErrchk( cudaMemcpy(im1_gpu, L[0], sizeof(unsigned int) * rows * cols, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(im2_gpu, R[0], sizeof(unsigned int) * rows * cols, cudaMemcpyHostToDevice) );
        
        device_shift_subtract_stack(im1_gpu,
        im2_gpu,
        shifted_images_gpu,
        rows, cols);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }
    auto clock_end_gpu = std::chrono::system_clock::now();
    unsigned int elapsed_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end_gpu-clock_start_gpu).count();
    float avg_time_gpu = ((float) elapsed_time_gpu / (float) NUM_ITERS);
    std::cout << "average gpu time: " << avg_time_gpu << std::endl;


    // float * shifted_images_cpu = new float[rows * cols * D];
    // gpuErrchk( cudaMemcpy(shifted_images_cpu, shifted_images_gpu, sizeof(float) * rows * cols * D, cudaMemcpyDeviceToHost) );


    // for(int i = 0; i < rows * cols * D; i++)
    // {
    //     assert(shifted_images_cpu[i] == shifted_images[0][i]);
    // }   
}
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "transpose_device.cuh"
#include "ta_utilities.hpp"
/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *         what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/* a and b point to n x n matrices. This method checks that A = B^T. */
void checkTransposed(const float *a, const float *b, int n) {
    bool correct = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i + n * j] != b[j + n * i]) {
                correct = false;
                fprintf(stderr,
                    "Transpose failed: a[%d, %d] != b[%d, %d], %f != %f\n",
                    i, j, j, i, a[i + n * j], b[j + n * i]);
                assert(correct);
            }
        }
    }

    assert(correct);
}

/* Naive CPU transpose, takes an n x n matrix in input and writes to output. */
void cpuTranspose(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            output[j + n * i] = input[i + n * j];
        }
    }
}

/*
 * Fills fill with random numbers is [0, 1]. Size is number of elements to
 * assign.
 */
void randomFill(float *fill, int size) {
    for (int i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        fill[i] = r;
    }
}

int main(int argc, char *argv[]) {

    // These functions allow you to select the least utilized GPU 
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these 
    // functions if you are running on your local machine.
    TA_Utilities::select_coldest_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    
    // Seed random number generator
    srand(2016);

    std::string kernel = "all";
    int size_to_run = -1;

    // Check arguments
    assert(argc <= 3);
    if (argc >= 2)
        size_to_run = atoi(argv[1]);
    if (argc == 3)
        kernel = argv[2];

    if (!(size_to_run == -1  ||
         size_to_run == 512  ||
         size_to_run == 1024 ||
         size_to_run == 2048 ||
         size_to_run == 4096))
    {
        fprintf(stderr,
            "Program only designed to run sizes 512, 1024, 2048, 4096\n");
    }

    assert(kernel == "all"     ||
        kernel == "cpu"        ||
        kernel == "gpu_memcpy" ||
        kernel == "naive"      ||
        kernel == "shmem"      ||
        kernel == "optimal");

    // Run the transpose implementations for all desired sizes (2^9 = 512, 
    // 2^12 = 4096)
    for (int _i = 9; _i < 13; _i++) {
        int n = 1 << _i;
        if (size_to_run != -1 && size_to_run != n)
            continue;

        assert(n % 64 == 0);

        cudaEvent_t start;
        cudaEvent_t stop;

#define START_TIMER() {                                                        \
            gpuErrChk(cudaEventCreate(&start));                                \
            gpuErrChk(cudaEventCreate(&stop));                                 \
            gpuErrChk(cudaEventRecord(start));                                 \
        }

#define STOP_RECORD_TIMER(name) {                                              \
            gpuErrChk(cudaEventRecord(stop));                                  \
            gpuErrChk(cudaEventSynchronize(stop));                             \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
            gpuErrChk(cudaEventDestroy(start));                                \
            gpuErrChk(cudaEventDestroy(stop));                                 \
        }

        // Initialize timers
        float cpu_ms = -1;
        float gpu_memcpy = -1;
        float naive_gpu_ms = -1;
        float shmem_gpu_ms = -1;
        float optimal_gpu_ms = -1;

        // Allocate host memory
        float *input = new float[n * n];
        float *output = new float[n * n];

        // Allocate device memory
        float *d_input;
        float *d_output;
        gpuErrChk(cudaMalloc(&d_input, n * n * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_output, n * n * sizeof(float)));

        // Initialize input data to random numbers in [0, 1]
        randomFill(input, n * n);

        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_input, input, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));


        // CPU implementation
        if (kernel == "cpu" || kernel == "all") {
            START_TIMER();
            cpuTranspose(input, output, n);
            STOP_RECORD_TIMER(cpu_ms);

            checkTransposed(input, output, n);
            memset(output, 0, n * n * sizeof(float));

            printf("Size %d naive CPU: %f ms\n", n, cpu_ms);
        }

        // GPU memcpy (useful for comparison purposes)
        if (kernel == "gpu_memcpy" || kernel == "all") {
            START_TIMER();
            gpuErrChk(cudaMemcpy(d_output, d_input, n * n * sizeof(float),
                cudaMemcpyDeviceToDevice));
            STOP_RECORD_TIMER(gpu_memcpy);

            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));
            printf("Size %d GPU memcpy: %f ms\n", n, gpu_memcpy);
        }

        // Naive GPU implementation
        if (kernel == "naive" || kernel == "all") {
            START_TIMER();
            cudaTranspose(d_input, d_output, n, NAIVE);
            STOP_RECORD_TIMER(naive_gpu_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * n * sizeof(float), 
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * n * sizeof(float));
            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

            printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        }

        // shmem GPU implementation
        if (kernel == "shmem" || kernel == "all") {
            START_TIMER();
            cudaTranspose(d_input, d_output, n, SHMEM);
            STOP_RECORD_TIMER(shmem_gpu_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * n * sizeof(float), 
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * n * sizeof(float));
            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

            printf("Size %d shmem GPU: %f ms\n", n, shmem_gpu_ms);
        }

        // Optimal GPU implementation
        if (kernel == "optimal"    || kernel == "all") {
            START_TIMER();
            cudaTranspose(d_input, d_output, n, OPTIMAL);
            STOP_RECORD_TIMER(optimal_gpu_ms);

            gpuErrChk(cudaMemcpy(output, d_output, n * n * sizeof(float), 
                cudaMemcpyDeviceToHost));
            checkTransposed(input, output, n);

            memset(output, 0, n * n * sizeof(float));
            gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

            printf("Size %d optimal GPU: %f ms\n", n, optimal_gpu_ms);
        }

        // Free host memory
        delete[] input;
        delete[] output;

        // Free device memory
        gpuErrChk(cudaFree(d_input));
        gpuErrChk(cudaFree(d_output));

        printf("\n");
    }
}

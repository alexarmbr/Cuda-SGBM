#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "fft_convolve.cuh"
#include <cufft.h>

#include "ta_utilities.hpp"

using std::cerr;
using std::cout;
using std::endl;

const float PI = 3.14159265358979;

#if AUDIO_ON
    #include <sndfile.h>
#endif


float gaussian(float x, float mean, float std){
    return (1 / (std * sqrt(2 * PI) ) ) 
        * exp(-1.0/2.0 * pow((x - mean) / std, 2) );
}

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        cerr << "Failed FFT call, error code " << errval << endl;
    }
}


void check_args(int argc, char **argv){

#if AUDIO_ON
    if (argc != 6){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <max number of blocks> <input file> <impulse response file> <output file>\n";
        exit(EXIT_FAILURE);
    }
#else
    if (argc != 3){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <max number of blocks>\n";
        exit(EXIT_FAILURE);
    }
#endif
}


/* Reads in audio data (alternatively, generates random data), 
and convolves each channel with the specified 
filtering function h[n], producing output data. 

Uses both CPU and GPU implementations, and compares the results.
*/

int large_gauss_test(int argc, char **argv){

    check_args(argc, argv);



/*************** GAUSSIAN ***************/

    /* Form Gaussian blur vector */
    float mean = 0.0;
    float std = 500.0;

    int GAUSSIAN_SIDE_WIDTH = 1000;
    int GAUSSIAN_SIZE = 2 * GAUSSIAN_SIDE_WIDTH + 1;

    // Space for both sides of the gaussian blur vector, plus the middle,
    // gives this size requirement
    float *gaussian_v = (float*)malloc(sizeof(float) * GAUSSIAN_SIZE );

    // Fill it from the middle out
    for (int i = -GAUSSIAN_SIDE_WIDTH; i <= GAUSSIAN_SIDE_WIDTH; i++){

        gaussian_v[ GAUSSIAN_SIDE_WIDTH + i ] = gaussian(i, mean, std);

    }

/*************** END GAUSSIAN ******************/






#if AUDIO_ON

    SNDFILE *in_file, *impulse_file, *out_file;
    SF_INFO in_file_info, impulse_file_info, out_file_info;

    int amt_read_input;
    int amt_read_impulse;



    // Open input audio file
    in_file = sf_open(argv[3], SFM_READ, &in_file_info);
    if (!in_file){
        cerr << "Cannot open input file, exiting\n";
        exit(EXIT_FAILURE);
    }

    // Read input
    float *allchannel_input = new float[in_file_info.frames * in_file_info.channels];
    amt_read_input = sf_read_float(in_file, allchannel_input, 
        in_file_info.frames * in_file_info.channels);
    assert(amt_read_input == in_file_info.frames * in_file_info.channels);






    // Open impulse response audio file
    impulse_file = sf_open(argv[4], SFM_READ, &impulse_file_info);
    if (!impulse_file){
        cerr << "Cannot open impulse response put file, exiting\n";
        exit(EXIT_FAILURE);
    }

    // Read impulse response
    float *allchannel_impulse = new float[impulse_file_info.frames * impulse_file_info.channels];
    amt_read_impulse = sf_read_float(impulse_file, allchannel_impulse, 
        impulse_file_info.frames * impulse_file_info.channels);
    assert(amt_read_impulse == impulse_file_info.frames * impulse_file_info.channels);








    // For now, have same number of channels in both files
    assert(impulse_file_info.channels == in_file_info.channels);

    int nChannels = in_file_info.channels;

    int N = in_file_info.frames;
    int impulse_length = impulse_file_info.frames;


#else

    /* If we're using random data instead of audio data, we can
    control the size of our input signal, and use the "channels"
    parameter to control how many trials we run. */

    int nChannels = 2;      // Can set as the number of trials
    int N = 1e7;        // Can set how many data points arbitrarily
    int impulse_length = GAUSSIAN_SIZE;

#endif










    cout << endl << "N (number of samples per channel):    " << N << endl;
    cout << endl << "Impulse length (number of samples per channel):    " 
        << impulse_length << endl;


    /* This is the minimum length of x and h required for FFT convolution
    to function and to emulate linear convolution with circular convolution. */
    int padded_length = N + impulse_length - 1;

    // Per-channel input data
    cufftComplex *input_data
        = (cufftComplex*)malloc(sizeof(cufftComplex) * N);

    // Per-channel output data
    cufftComplex *output_data
        = (cufftComplex*)malloc(sizeof(cufftComplex) * padded_length);


    // Per-channel impulse response data (change the name later)
    cufftComplex *impulse_data
        = (cufftComplex*)malloc(sizeof(cufftComplex) * impulse_length);


    // Per-channel un-normalized output buffer
    // Used for testing and timing purposes only (normalization check)
    cufftComplex *output_data_testarr
        = (cufftComplex*)malloc(sizeof(cufftComplex) * padded_length);


#if AUDIO_ON

    // Prepare output storage
    float *allchannel_output = new float[padded_length * nChannels];

#endif





    // Output storage for CPU implementation
    float *output_data_host = (float*)malloc(padded_length * sizeof(float));


    // GPU copies of arrays
    cufftComplex *dev_input_data;
    cufftComplex *dev_impulse_v;
    cufftComplex *dev_out_data;

    /* TODO: Allocate memory for these three arrays on the GPU. 

    Note that we need to allocate more memory than on Homework 1,
    due to the padding necessary for the FFT convolution. 

    Also, unlike in Homework 1, we don't copy our impulse response
    yet, because this is now given to us per-channel. */





// (From Eric's code)
    cudaEvent_t start;
    cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
    }




    /* Iterate through each audio channel (e.g. 2 iterations for 
    stereo files) */
    for (int ch = 0; ch < nChannels; ch++){



    #if AUDIO_ON

        // Load this channel's input data
        for (int i = 0; i < N; i++){
            input_data[i].x = allchannel_input[ (i * nChannels) + ch ];
            input_data[i].y = 0;
        }

        // Load this channel's impulse response data
        for (int i = 0; i < impulse_length; i++){
            impulse_data[i].x = allchannel_impulse[ (i * nChannels) + ch ];
            impulse_data[i].y = 0;
        }



    #else
        // Generate random data if not using audio
        for (int i = 0; i < N; i++){
            input_data[i].x = ((float)rand() ) / RAND_MAX;
            input_data[i].y = 0;
        }


        // Use gaussian kernel if not using audio (same both channels)
        for (int i = 0; i < impulse_length; i++){
            impulse_data[i].x = gaussian_v[i];
            impulse_data[i].y = 0;
        }

    #endif



        float cpu_time_ms_convolve = -1;
        float gpu_time_ms_convolve = -1;

        float cpu_time_ms_norm = -1;
        float gpu_time_ms_norm = -1;


        /* CPU Blurring */


        cout << endl << "CPU convolution..." << endl;

        memset(output_data_host, 0, padded_length * sizeof(float));

        // Use the CUDA machinery for recording time
        START_TIMER();

        // (For scoping)

        {
            #if 1

            for (int i = 0; i < impulse_length; i++){
                for (int j = 0; j <= i; j++){
                    output_data_host[i] += input_data[i - j].x 
                                            * impulse_data[j].x; 
                }
            }
            for (int i = impulse_length; i < N; i++){
                for (int j = 0; j < impulse_length; j++){
                    output_data_host[i] += input_data[i - j].x 
                                            * impulse_data[j].x; 
                }
            }
            for (int i = N; i < padded_length; i++){
                for (int j = i - (N - 1); j < impulse_length; j++){
                    output_data_host[i] += input_data[i - j].x 
                                            * impulse_data[j].x; 
                }
            }

            #endif
        }

        STOP_RECORD_TIMER(cpu_time_ms_convolve);






        /* GPU convolution */

        cout << "GPU convolution..." << endl;


        // Cap the number of blocks
        const unsigned int local_size = atoi(argv[1]);
        const unsigned int max_blocks = atoi(argv[2]);
        const unsigned int blocks = std::min( max_blocks, 
            (unsigned int) ceil(N/(float)local_size) );



        // Start timer...
        START_TIMER();



        /* TODO: Copy this channel's input data (stored in input_data)
        from host memory to the GPU. 

        Note that input_data only stores
        x[n] as read from the input audio file, and not the padding, 
        so be careful with the size of your memory copy. */




        /* TODO: Copy this channel's impulse response data (stored in impulse_data)
        from host memory to the GPU. 

        Like input_data, impulse_data
        only stores h[n] as read from the impulse response file, 
        and not the padding, so again, be careful with the size
        of your memory copy. (It's not the same size as the input_data copy.)
        */


        /* TODO: We're only copying to part of the allocated
        memory regions on the GPU above, and we still need to zero-pad.
        (See Lecture 9 for details on padding.)
        Set the rest of the memory regions to 0 (recommend using cudaMemset).
        */


        /* TODO: Create a cuFFT plan for the forward and inverse transforms. 
        (You can use the same plan for both, as is done in the lecture examples.)
        */


        /* TODO: Run the forward DFT on the input signal and the impulse response. 
        (Do these in-place.) */




        /* NOTE: This is a function in the fft_convolve_cuda.cu file,
        where you'll fill in the kernel call for point-wise multiplication
        and scaling. */
        cudaCallProdScaleKernel(blocks, local_size, 
            dev_input_data, dev_impulse_v, dev_out_data,
            padded_length);


        // Check for errors on kernel call
        cudaError err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
        } else {
                cerr << "No kernel error detected" << endl;
        }






        /* TODO: Run the inverse DFT on the output signal. 
        (Do this in-place.) */



        /* TODO: Destroy the cuFFT plan. */


        // For testing and timing-control purposes only
        gpuErrchk(cudaMemcpy(output_data_testarr, dev_out_data, padded_length * sizeof(cufftComplex), cudaMemcpyDeviceToHost));


        STOP_RECORD_TIMER(gpu_time_ms_convolve);





        cout << "Comparing..." << endl;

#if 1
        // Compare results
        // NOTE: Not comparing through the padded length
        // MAY NEED TO CHANGE THIS
        bool success = true;
        for (int i = 0; i < padded_length; i++){
            if (fabs(output_data_host[i] - output_data_testarr[i].x) < 1e-3){
                #if 0
                cout << "Correct output at index " << i << ": " << output_data_host[i] << ", " 
                    << output_data_testarr[i].x << endl;
                #endif
            } else {
                success = false;
                cerr << "Incorrect output at index " << i << ": " << output_data_host[i] << ", " 
                    << output_data_testarr[i].x << endl;
            }
        }

        if (success){
            cout << endl << "Successful output" << endl;
        }
#endif


        cout << endl;
        cout << "CPU time (convolve): " << cpu_time_ms_convolve << " milliseconds" << endl;
        cout << "GPU time (convolve): " << gpu_time_ms_convolve << " milliseconds" << endl;
        cout << endl << "Speedup factor (convolution): " << cpu_time_ms_convolve / gpu_time_ms_convolve << endl << endl;

        // Normalize output (avoids clipping and/or hearing loss)
        // such that maximum value is < 1 (say 0.99999)


        /* CPU normalization... */

        cout << endl;
        cout << "CPU normalization..." << endl;

        START_TIMER();

        float max_abs_val = fabs(output_data_testarr[0].x);

        {

            float max_allowed_amp = 0.99999;


            for (int i = 1; i < padded_length; i++){
                float val = fabs(output_data_testarr[i].x);
                if (val > max_abs_val){
                    max_abs_val = val;
                }
            }

            for (int i = 0; i < padded_length; i++){
                output_data_testarr[i].x *= (max_allowed_amp / max_abs_val);
            }

        }

        STOP_RECORD_TIMER(cpu_time_ms_norm);




        /* GPU normalization... */


        cout << "GPU normalization..." << endl;

        START_TIMER();



        float max_abs_val_fromGPU;
        float *dev_max_abs_val;


        // Exclude these from timer since the only reason we have these is
        // for testing purposes
        // Also keep this in the student code
        // NVM they need this for global ops
        // Also need to memset to 0 for baseline value

        /* TODO 2: Allocate memory to store the maximum magnitude found. 
        (You only need enough space for one floating-point number.) */

        /* TODO 2: Set it to 0 in preparation for running. 
        (Recommend using cudaMemset) */


        /* NOTE: This is a function in the fft_convolve_cuda.cu file,
        where you'll fill in the kernel call for finding the maximum
        of the output signal. */
        cudaCallMaximumKernel(blocks, local_size, dev_out_data,
            dev_max_abs_val, padded_length);

        // Check for errors on kernel call
        err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
        } else {
                cerr << "No kernel error detected" << endl;
        }


        /* NOTE: This is a function in the fft_convolve_cuda.cu file,
        where you'll fill in the kernel call for dividing the output
        signal by the previously-calculated maximum. */
        cudaCallDivideKernel(blocks, local_size, dev_out_data,
            dev_max_abs_val, padded_length);

        // Check for errors on kernel call
        err = cudaGetLastError();
        if  (cudaSuccess != err){
                cerr << "Error " << cudaGetErrorString(err) << endl;
        } else {
                cerr << "No kernel error detected" << endl;
        }



        STOP_RECORD_TIMER(gpu_time_ms_norm);

        // For testing purposes only
        gpuErrchk( cudaMemcpy(&max_abs_val_fromGPU, 
            dev_max_abs_val, 1 * sizeof(float), cudaMemcpyDeviceToHost) );



        /* TODO: Now that kernel calls have finished, copy the output
        signal back from the GPU to host memory. (We store this channel's
        result in output_data on the host.)

        Note that we have a padded-length signal, so be careful of the
        size of the memory copy. */


        cout << endl;
        cout << "CPU normalization constant: " << max_abs_val << endl;
        cout << "GPU normalization constant: " << max_abs_val_fromGPU << endl;

        if ( fabs(max_abs_val_fromGPU - max_abs_val) > 1e-6 ){
            cerr << "!! Incorrect normalization" << endl;
        }

        cout << endl;
        cout << "CPU time (normalization): " << cpu_time_ms_norm << " milliseconds" << endl;
        cout << "GPU time (normalization): " << gpu_time_ms_norm << " milliseconds" << endl;

        cout << endl << "Speedup factor (normalization): " << cpu_time_ms_norm / gpu_time_ms_norm << endl << endl;
        cout << endl << endl;

        // Write output audio data to multichannel array
        #if AUDIO_ON
            for (int i = 0; i < padded_length; i++){
                allchannel_output[i * nChannels + ch] = output_data[i].x;
            }
        #endif



    }



    // Free memory on GPU
    cudaFree(dev_input_data);
    cudaFree(dev_impulse_v);
    cudaFree(dev_out_data);

    // Free memory on host
    free(input_data);
    free(output_data);
    free(output_data_host);


// Write audio output to file
#if AUDIO_ON
    out_file_info = in_file_info;
    out_file = sf_open(argv[5], SFM_WRITE, &out_file_info);
    if (!out_file){
        cerr << "Cannot open output file, exiting\n";
        exit(EXIT_FAILURE);
    }

    sf_write_float(out_file, allchannel_output, nChannels * padded_length); 
    sf_close(in_file);
    sf_close(out_file);

#endif

    return EXIT_SUCCESS;

}


int main(int argc, char **argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_coldest_GPU();
    int max_time_allowed_in_seconds = 90;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    return large_gauss_test(argc, argv);
}



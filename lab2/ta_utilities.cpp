//--------------------------------------------------------------------------
// TA_Utilities.cpp
// Allow a shared computer to run smoothly when it is being used
// by students in a CUDA GPU programming course.
//
// TA_Utilities.cpp/hpp provide functions that programatically limit
// the execution time of the function and select the GPU with the 
// lowest temperature to use for kernel calls.
//
// Author: Jordan Bonilla - 4/6/16
//--------------------------------------------------------------------------

#include "ta_utilities.hpp"

#include <unistd.h> // sleep, fork, getpid
#include <signal.h> // kill
#include <cstdio> // printf
#include <stdlib.h> // popen, pclose, atoi, fread
#include <cuda_runtime.h> // cudaGetDeviceCount, cudaSetDevice

namespace TA_Utilities
{
  /* Select the least utilized GPU on this system. Estimate
     GPU utilization using GPU temperature. UNIX only. */
  void select_coldest_GPU() 
  {
      // Get the number of GPUs on this machine
      int num_devices;
      cudaGetDeviceCount(&num_devices);
      if(num_devices == 0) {
          printf("select_coldest_GPU: Error - No GPU detected\n");
          return;
      }
      // Read GPU info into buffer "output"
      const unsigned int MAX_BYTES = 10000;
      char output[MAX_BYTES];
      FILE *fp = popen("nvidia-smi &> /dev/null", "r");
      size_t bytes_read = fread(output, sizeof(char), MAX_BYTES, fp);
      pclose(fp);
      if(bytes_read == 0) {
          printf("Error - No Temperature could be read\n");
          return;
	  }
      // array to hold GPU temperatures
      int * temperatures = new int[num_devices];
      // parse output for temperatures using knowledge of "nvidia-smi" output format
      int i = 0;
      unsigned int num_temps_parsed = 0;
      while(output[i] != '\0') {
          if(output[i] == '%') {
              unsigned int temp_begin = i + 1;
              while(output[i] != 'C') {
                  ++i;
              }
              unsigned int temp_end = i;
              char this_temperature[32];
              // Read in the characters cooresponding to this temperature
              for(unsigned int j = 0; j < temp_end - temp_begin; ++j) {
                  this_temperature[j] = output[temp_begin + j];
              }
              this_temperature[temp_end - temp_begin + 1] = '\0';
              // Convert string representation to int
              temperatures[num_temps_parsed] = atoi(this_temperature);
              num_temps_parsed++;
          }
          ++i;
      }
      // Get GPU with lowest temperature
      int min_temp = 1e7, index_of_min = -1;
      for (int i = 0; i < num_devices; i++) 
      {
          int candidate_min = temperatures[i];
          if(candidate_min < min_temp) 
          {
              min_temp = candidate_min;
              index_of_min = i;
          }
      }
      // Tell CUDA to use the GPU with the lowest temeprature
      printf("Index of the GPU with the lowest temperature: %d (%d C)\n", 
          index_of_min, min_temp);
      cudaSetDevice(index_of_min);
      // Free memory and return
      delete(temperatures);
      return;
  } // end "void select_coldest_GPU()""

  /* Create a child thread that will kill the parent thread after the
     specified time limit has been exceeded */
  void enforce_time_limit(int time_limit) {
      printf("Time limit for this program set to %d seconds\n", time_limit);
      int parent_id = getpid();
      pid_t child_id = fork();
      // The fork call creates a lignering child thread that will 
      // kill the parent thread after the time limit has exceeded
      // If it hasn't already terminated.
      if(child_id == 0) // "I am the child thread"
      {
          sleep(time_limit);
          if( kill(parent_id, SIGTERM) == 0) {
              printf("enforce_time_limit.c: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          }
          // Ensure that parent was actually terminated
          sleep(2);
          if( kill(parent_id, SIGKILL) == 0) {
              printf("enforce_time_limit.c: Program terminated"
               " for taking longer than %d seconds\n", time_limit);
          } 
          // Child thread has done its job. Terminate now.
          exit(0);
      }
      else // "I am the parent thread"
      {
          // Allow the parent thread to continue doing what it was doing
          return;
      }
  } // end "void enforce_time_limit(int time_limit)


} // end "namespace TA_Utilities"

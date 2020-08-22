//--------------------------------------------------------------------------
// TA_Utilities.hpp
// Allow a shared computer to run smoothly when it is being used
// by students in a CUDA GPU programming course.
//
// TA_Utilities.cpp/hpp provide functions that programatically limit
// the execution time of the function and select the GPU with the 
// lowest temperature to use for kernel calls.
//--------------------------------------------------------------------------

#pragma once

namespace TA_Utilities
{
    /* Create a child thread that will kill the parent thread after the
       specified time limit has been exceeded. UNIX only */
    void enforce_time_limit(int time_limit);
    /* Select the least utilized GPU on this system. Estimate
       GPU utilization using GPU temperature. UNIX only. */
    void select_coldest_GPU();

}

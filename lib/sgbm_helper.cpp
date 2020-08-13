
#include <iostream>
using namespace std;

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


extern "C" void shift_subtract_stack(unsigned long long int * L,
unsigned long long int R,
unsigned long long int out,
int rows, int cols, int csize){

    int d = -1;
    for(int i = 0; i < rows * cols * d; i++){

        if (i % rows * cols == 0)
            d+=1;

        if (i % cols > cols - 1 - d){
            

        }
    }








}






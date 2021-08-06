#include "lib/sgbm_helper.hpp"

#include <iostream>
#include <chrono>
#include <random>
#include <limits>

unsigned long long * genData(int size)
{

    unsigned long long * arr = new unsigned long long [size];
    std::random_device rd;     
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long long> distr;

    for(int i = 0; i < size; i++)
        arr[i] = distr(eng);
    return arr;
}

int main( int argc, char** argv )
{

    int rows = 433;
    int cols = 1234;
    unsigned long long * L = genData(rows * cols);
    unsigned long long * R = genData(rows * cols);
    float * shifted_images = new float [nCols * nRows * D];

    shift_subtract_stack(L,R,shifted_images, rows, cols);



}
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <omp.h>

#include "lib/sgbm_helper.hpp"
#include "assert.h"
#include "dirent.h"

const int NUM_IMAGES = 3;

int main( int argc, char** argv )
{
    const char * path = "2010_03_09_drive_0023/2010_03_09_drive_0023_Images";
    std::vector<std::string> leftIm;
    std::vector<std::string> rightIm;
    
    DIR *dir;
    struct dirent *ent;
    
    if ((dir = opendir (path)) == NULL)
        return -1;

    while ((ent = readdir (dir)) != NULL)
    {
        std::string entry = ent->d_name;
        if (entry[1] == '1')
            leftIm.push_back(entry);
        else if(entry[1] == '2')
            rightIm.push_back(entry);
    }
    std::sort(leftIm.begin(), leftIm.end());
    std::sort(rightIm.begin(), rightIm.end());
    assert(leftIm.size() == rightIm.size());
    
    cv::Mat im1;
    im1 = imread((std::string) path + "/" + leftIm[0], cv::IMREAD_GRAYSCALE);
    int nRows = im1.rows;
    int nCols = im1.cols;

    int numImages = std::min(NUM_IMAGES,  (int) leftIm.size());
    auto clock_start = std::chrono::system_clock::now();
    
    #pragma omp parallel
    {
        cv::Mat im1;
        cv::Mat im2;
        int * depth_im = new int[nCols * nRows];

        #pragma omp parallel for
        for(int i=0; i < numImages; i++)
        {
            int tid = omp_get_thread_num();
            im1 = imread((std::string) path + "/" + leftIm[i], cv::IMREAD_GRAYSCALE);
            im2 = imread((std::string) path + "/" + rightIm[i], cv::IMREAD_GRAYSCALE);
            _sgbm(&im1, &im2, depth_im, nRows, nCols, tid);
        }
    }
    auto clock_end = std::chrono::system_clock::now();
    unsigned int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end-clock_start).count();
    std::cout << "processing "  << numImages <<  " from disk took " << elapsed_time << " ms" << std::endl;
    float fps = ((float) numImages / (float) elapsed_time) * 1000;
    std::cout << "at " << fps << " fps" << std::endl;

}
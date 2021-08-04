#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <vector>

#include "assert.h"


namespace fs = std::filesystem;
int main( int argc, char** argv )
{
    std::string path = "2010_03_09_drive_0023/2010_03_09_drive_0023_Images";
    std::vector<std::string> leftIm;
    std::vector<std::string> rightIm;
    
    for (const auto & fileNameObj : fs::directory_iterator(path))
    {
        std::string entry = fileNameObj.path();
        if (entry[52] == '1')
            leftIm.push_back(entry);
        else if(entry[52] == '2')
            rightIm.push_back(entry);
    }
    std::sort(leftIm.begin(), leftIm.end());
    std::sort(rightIm.begin(), rightIm.end());
    assert(leftIm.size() == rightIm.size());

    cv::Mat im1;
    cv::Mat im2;
    
    auto clock_start = std::chrono::system_clock::now();
    for(int i=0; i < leftIm.size(); i++)
    {
        im1 = imread(leftIm[i], cv::IMREAD_GRAYSCALE);
        im2 = imread(rightIm[i], cv::IMREAD_GRAYSCALE);
    }
    auto clock_end = std::chrono::system_clock::now();
    unsigned int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end-clock_start).count();
    std::cout << "reading all those images from disk took " << elapsed_time << " ms" << std::endl;
    float fps = ((float) leftIm.size() / (float) elapsed_time) * 1000;
    std::cout << "at " << fps << " fps" << std::endl;

}
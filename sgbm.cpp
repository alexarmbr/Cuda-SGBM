#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lib/sgbm_helper.hpp"

//#include "lib/sgbm_helper.cuh"
#include <iostream>
#include <chrono>


using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // if( argc != 3)
    // {
    //  cout <<" Usage: sgbm Image1 Image2" << endl;
    //  return -1;
    // }

    Mat im1;
    Mat im2;
    im1 = imread(argv[1], IMREAD_GRAYSCALE);
    im2 = imread(argv[2], IMREAD_GRAYSCALE);
    // im1 = imread("im1.png", IMREAD_GRAYSCALE);
    // im2 = imread("im0.png", IMREAD_GRAYSCALE);
    //resize(im1, im1, Size(480,480));
    //resize(im2, im2, Size(480,480));

    if(! im1.data | ! im2.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    CV_Assert(im1.depth() == CV_8U);
    int nRows = im1.rows;
    int nCols = im1.cols;
    std::cout << nRows << std::endl;
    std::cout << nCols << std::endl;
    int * depth_im = new int[nCols * nRows];
    _sgbm(&im1, &im2, depth_im, nRows, nCols);
    FILE *f = fopen("stereo_im.data", "wb");
    fwrite(depth_im, sizeof(int), nCols * nRows, f);
    fclose(f);
    return 0;
}
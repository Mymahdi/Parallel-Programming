#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


__constant__ float sobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ float sobelY[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


int main() {
    cv::Mat image = cv::imread("images/Lenna.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imgSize = width * height * sizeof(unsigned char);

    unsigned char* h_input = image.data;
    unsigned char* h_output = (unsigned char*)malloc(imgSize);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

    std::cout << "Image loaded successfully!" << std::endl;
    return 0;
}

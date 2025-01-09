#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


__global__ void sobelFilter(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ unsigned char sharedMem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Load input to shared memory
    if (x < width && y < height) {
        sharedMem[threadIdx.y + 1][threadIdx.x + 1] = input[y * width + x];
        if (threadIdx.x == 0 && x > 0) // Left halo
            sharedMem[threadIdx.y + 1][0] = input[y * width + x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1) // Right halo
            sharedMem[threadIdx.y + 1][BLOCK_SIZE + 1] = input[y * width + x + 1];
        if (threadIdx.y == 0 && y > 0) // Top halo
            sharedMem[0][threadIdx.x + 1] = input[(y - 1) * width + x];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1) // Bottom halo
            sharedMem[BLOCK_SIZE + 1][threadIdx.x + 1] = input[(y + 1) * width + x];
    }
    __syncthreads();

    if (x < width && y < height) {
        float gradX = 0.0f;
        float gradY = 0.0f;

        #pragma unroll
        for (int ky = -1; ky <= 1; ++ky) {
            #pragma unroll
            for (int kx = -1; kx <= 1; ++kx) {
                gradX += sobelX[ky + 1][kx + 1] * sharedMem[threadIdx.y + 1 + ky][threadIdx.x + 1 + kx];
                gradY += sobelY[ky + 1][kx + 1] * sharedMem[threadIdx.y + 1 + ky][threadIdx.x + 1 + kx];
            }
        }

        output[y * width + x] = min(255, (int)sqrtf(gradX * gradX + gradY * gradY));
    }
}



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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobelFilter<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    cv::Mat outputImage(height, width, CV_8UC1, h_output);
    cv::imwrite("output.jpg", outputImage);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    std::cout << "Image loaded successfully!" << std::endl;
    return 0;
}


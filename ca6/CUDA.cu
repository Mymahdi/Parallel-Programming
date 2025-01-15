#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

typedef unsigned char byte;

void toGreyScale(byte *input, byte *output, int h, int w, int ch) {
    int i, j;
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            int ind = i * w * ch + j * ch;
            byte res = input[ind + 0] * 0.2989 + input[ind + 1] * 0.5870 + input[ind + 2] * 0.1140;
            output[i * w + j] = res;
        }
    }
}

// Apply the Sobel edge detection kernel
void applySobel(byte *input, byte *output, int h, int w) {
    int x, y;
    byte *img = input;
    
    // Sobel kernel for detecting edges in horizontal and vertical directions
    int kernel_x[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    int kernel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (y = 1; y < h - 1; y++) {
        for (x = 1; x < w - 1; x++) {
            int gx = 0, gy = 0;

            // Apply the Sobel kernels
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = img[(y + ky) * w + (x + kx)];
                    gx += kernel_x[ky + 1][kx + 1] * pixel;
                    gy += kernel_y[ky + 1][kx + 1] * pixel;
                }
            }

            // Calculate the gradient magnitude
            int gradient = (int)sqrt(gx * gx + gy * gy);
            gradient = gradient > 255 ? 255 : (gradient < 0 ? 0 : gradient);

            // Store the result
            output[y * w + x] = (byte)gradient;
        }
    }
}

int main() {

    // Load sample image
    cv::Mat inputImage = cv::imread("images/flower.png", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        printf("Failed to load image!\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    byte *input = inputImage.data;
    byte *grayImage = (byte *)malloc(width * height * sizeof(byte));
    byte *outputImage = (byte *)malloc(width * height * sizeof(byte));

    // Convert to grayscale
    toGreyScale(input, grayImage, height, width, channels);

    // Apply Sobel edge detection
    applySobel(grayImage, outputImage, height, width);

    // Convert the result back to a Mat object for visualization
    cv::Mat result(height, width, CV_8UC1, outputImage);

    // Normalize the result to enhance the edges
    cv::Mat enhancedResult;
    cv::normalize(result, enhancedResult, 0, 255, cv::NORM_MINMAX);

    // Save the result as an image
    cv::imwrite("output_sobel.jpg", enhancedResult);

    // Display the result
    cv::imshow("Edge Detection Result", enhancedResult);
    cv::waitKey(0);

    free(grayImage);
    free(outputImage);
    printf("Edge-detected image saved as output_sobel.jpg\n");

    return 0;
}

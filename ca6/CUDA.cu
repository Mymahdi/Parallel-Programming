#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <float.h>
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


void prewitt(byte *input, byte *output, int h, int w) {
    int x, y;
    byte *img = input;
    for (y = 1; y < h - 1; y++) {
        for (x = 1; x < w - 1; x++) {
            int vKer = 0, hKer = 0;

            vKer = img[(y-1)*w+(x-1)] * -1 + img[(y-1)*w+x] * -1 + img[(y-1)*w+(x+1)] * -1 +
                   img[(y+1)*w+(x-1)] *  1 + img[(y+1)*w+x] *  1 + img[(y+1)*w+(x+1)] *  1;

            hKer = img[(y-1)*w+(x-1)] * -1 + img[(y-1)*w+(x+1)] *  1 +
                   img[y*w+(x-1)] * -1 + img[y*w+(x+1)] *  1 +
                   img[(y+1)*w+(x-1)] * -1 + img[(y+1)*w+(x+1)] *  1;

            int gradient = (int)sqrt(hKer * hKer + vKer * vKer);
            gradient = gradient > 255 ? 255 : gradient;

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

    return 0;
}




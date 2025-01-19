#define BLOCK_SIZE 16

__global__ void applyKernel(const float* input, float* output, int padded_height, int padded_width, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int image_height = padded_height - 2;
    int image_width = padded_width - 2;

    if (x >= image_width || y >= image_height) return;

    float kernel[9] = {
         0,  1,  0,
         1, -2,  1,
         0,  1,  0
    };

    for (int c = 0; c < channels; ++c) {
        float value = 0.0f;

        value += input[((y + 0) * padded_width + (x + 0)) * channels + c] * kernel[0];
        value += input[((y + 0) * padded_width + (x + 1)) * channels + c] * kernel[1];
        value += input[((y + 0) * padded_width + (x + 2)) * channels + c] * kernel[2];
        value += input[((y + 1) * padded_width + (x + 0)) * channels + c] * kernel[3];
        value += input[((y + 1) * padded_width + (x + 1)) * channels + c] * kernel[4];
        value += input[((y + 1) * padded_width + (x + 2)) * channels + c] * kernel[5];
        value += input[((y + 2) * padded_width + (x + 0)) * channels + c] * kernel[6];
        value += input[((y + 2) * padded_width + (x + 1)) * channels + c] * kernel[7];
        value += input[((y + 2) * padded_width + (x + 2)) * channels + c] * kernel[8];

        output[(y * image_width + x) * channels + c] = value;
    }
}

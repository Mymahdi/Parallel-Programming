#include <math.h>
#include <stdlib.h>
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

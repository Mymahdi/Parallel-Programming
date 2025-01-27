# CUDA Kernel Implementation Overview

This project implements a CUDA kernel for image processing, applying a convolution kernel for edge detection. The convolution kernel used is:


```
[ 0,  1,  0 ]
[ 1, -2,  1 ]
[ 0,  1,  0 ]
```

The workflow includes loading an image into device memory, processing it on the GPU, and saving the processed output to the host. Key features include efficient memory allocation, optimized kernel execution using CUDA threads, and loop unrolling for performance.

## Workflow Summary

1. **Image Loading**: Use OpenCV to read and validate the input image.
2. **Device Memory Management**: Allocate GPU memory for the input and output images using `cudaMalloc`.
3. **Data Transfer**: Copy the input image from host to device memory with `cudaMemcpy`.
4. **Kernel Execution**: Launch CUDA threads with a grid and block configuration tailored for optimal GPU performance. Each thread processes one pixel using the convolution kernel.
5. **Boundary Checks**: Ensure threads outside valid image dimensions skip computation to prevent memory errors.
6. **Performance Optimizations**: Use loop unrolling for efficient convolution and avoid warp divergence for improved parallelism.
7. **Data Retrieval**: Transfer the processed image back to the host using `cudaMemcpy`.
8. **Cleanup**: Free device memory with `cudaFree` and save the output image.

## Highlights

- **Speedup**: Achieved significant performance improvement compared to sequential processing, with execution time reduced to ~1 ms.
- **Efficiency**: Optimized GPU resource utilization through careful kernel design and memory access patterns.
- **Practical Use**: The output image demonstrates effective edge detection.

This project showcases how CUDA enables high-performance image processing by leveraging GPU parallelism for computationally intensive tasks.

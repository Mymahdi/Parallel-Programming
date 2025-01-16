import numpy as np
import argparse
from PIL import Image  # For loading and saving images

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    result = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                region = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                result[y, x, c] = np.sum(region * kernel)

    return result

def apply_kernel(slice_info):
    image_slice, kernel, top_overlap, bottom_overlap = slice_info
    processed_slice = convolve(image_slice, kernel)

    if top_overlap > 0:
        processed_slice = processed_slice[top_overlap:]
    if bottom_overlap > 0:
        processed_slice = processed_slice[:-bottom_overlap]

    return processed_slice

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply a convolution kernel to an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image.")
    args = parser.parse_args()

    # Load the image
    image = np.array(Image.open(args.image_path))

    kernel = np.array([
        [0, 1, 0],
        [1, -2, 1],
        [0, 1, 0]
    ])

    # Apply the kernel to the image
    result = apply_kernel((image, kernel, 0, 0))

    # Save the result
    Image.fromarray(result).save(args.output)
    print(f"Processed image saved to {args.output}")

if __name__ == "__main__":
    main()
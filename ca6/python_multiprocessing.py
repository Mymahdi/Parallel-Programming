import numpy as np

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


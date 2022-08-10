import numpy as np
import matplotlib.pyplot as plt
import cv2

# implement a function that returns a gaussian kernel
def make_gaussian(size=3, std=1):
    # Your code here
    return kernel

# implement a 2D convolution
def convolve2d(image, kernel):
    # You do not need to modify these, but feel free to implement your own
    kernel       = np.flipud(np.fliplr(kernel))  # Flip the kernel, if it's symmetric it does not matter
    kernel_width = kernel.shape[0]
    kernel_height= kernel.shape[1]
    padding      = (kernel_width - 1)
    offset       = padding // 2
    output       = np.zeros_like(image)
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + padding, image.shape[1] + padding))   
    image_padded[offset:-offset, offset:-offset] = image

    # implement the convolution inside the inner for loop
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Convolution - Your code here
            continue
    return output
import cv2
import math
import numpy as np


def line_segmentation(binary_image, gray_image):
    # Apply Dilation to mix all line words together
    dilation_kernel = np.ones((1, 190), np.uint8)
    image_dilation = cv2.dilate(np.invert(binary_image), dilation_kernel, iterations=1).astype(np.uint8)
    # Remove thin vertical lines to distinct overlaped lines 
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (370,1))
    remove_vertical = cv2.morphologyEx(image_dilation, cv2.MORPH_OPEN, vertical_kernel)
    # Find image contours which indicate lines 
    lines_contours, _ = cv2.findContours(remove_vertical.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extract lines
    binary_lines = []
    gray_lines = []
    for line in lines_contours:
        x, y, w, h = cv2.boundingRect(line)
        if h > 30:
            binary_lines.append(binary_image[max(y-30, 0):min(y+h+60, binary_image.shape[1]), x:x+w])
            gray_lines.append(gray_image[max(y-30, 0):min(y+h+60, gray_image.shape[1]), x:x+w])
    
    return binary_lines, gray_lines


def create_kernel(kernel_size, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernel_size % 2  # must be odd size
    half_size = kernel_size // 2

    kernel = np.zeros([kernel_size, kernel_size])
    sigma_x = sigma
    sigma_y = sigma * theta

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - half_size
            y = j - half_size

            exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
            x_term = (x ** 2 - sigma_x ** 2) / (2 * math.pi * sigma_x ** 5 * sigma_y)
            y_term = (y ** 2 - sigma_y ** 2) / (2 * math.pi * sigma_y ** 5 * sigma_x)

            kernel[i, j] = (x_term + y_term) * exp_term

    kernel = kernel / np.sum(kernel)
    return kernel


def word_segmentation(img, kernel_size=25, sigma=11, theta=7):
    kernel = create_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    _, binary_image = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    words_contours, _ = cv2.findContours(np.invert(binary_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return words_contours

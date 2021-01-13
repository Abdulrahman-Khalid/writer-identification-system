import numpy as np
import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern

def get_features(gray_images, binary_images, radius=3, no_points= 3*8, method='uniform'): # try no_points = 3 * 8
    features = []
    hist = np.zeros(256)
    for idx in range(len(binary_images)):
        lbp = local_binary_pattern(gray_images[idx], no_points, radius, method=method).astype(np.uint8)
        print(lbp)
        hist = cv2.calcHist([lbp], [0], binary_images[idx], [256], [0, 256], hist, True).ravel()
    hist /= np.mean(hist)
    features.extend(hist)

    return features



def is_bigger_than_center(gray_image, center, x, y):
    if x < 0 or y < 0 or x >= gray_image.shape[0] or y >= gray_image.shape[1]:
        return 0
    return 1 if gray_image[x][y] >= center else 0

def lbp_pixel(gray_image, x, y, radius=3, power_of_2=[1, 2, 4, 8, 16, 32, 64, 128]):
    """
    -------------------------------------------
    |  73 |  -  |  -  |  10 |  -  |  -  |  50 |
    -------------------------------------------
    |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
    -------------------------------------------
    |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
    -------------------------------------------
    |  0  |  -  |  -  | 50  |  -  |  -  |  40 |
    -------------------------------------------
    |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
    -------------------------------------------
    |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
    -------------------------------------------
    |  20 |  -  |  -  |  50 |  -  |  -  | 52  |
    -------------------------------------------
    """
    center = gray_image[x][y]
    pattern = 0
    pattern |= is_bigger_than_center(gray_image, center, x-radius, y-radius) * power_of_2[-1]      # top left
    pattern |= is_bigger_than_center(gray_image, center, x-radius, y) * power_of_2[-2]             # top
    pattern |= is_bigger_than_center(gray_image, center, x-radius, y+radius) * power_of_2[-3]      # top right
    pattern |= is_bigger_than_center(gray_image, center, x, y+radius) * power_of_2[-4]             # right
    pattern |= is_bigger_than_center(gray_image, center, x+radius, y+radius) * power_of_2[-5]      # bottom_right
    pattern |= is_bigger_than_center(gray_image, center, x+radius, y) * power_of_2[-6]             # bottom
    pattern |= is_bigger_than_center(gray_image, center, x+radius, y-radius) * power_of_2[-7]      # bottom_left
    pattern |= is_bigger_than_center(gray_image, center, x, y-radius) * power_of_2[-8]             # left
    return pattern

def lbp_features(gray_images, binary_image, radius=3):
    lbp_features = []
    hist = np.zeros(256)
    power_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]

    for idx in range(len(binary_images)):
        lbp_image = np.zeros(gray_images[idx].shape, np.uint8)
        for i in range(gray_images[idx].shape[0]):
            for j in range(gray_images[idx].shape[1]):
                lbp_image[i, j] = lbp_pixel(gray_images[idx], i, j, radius, power_of_2)
        print(idx, ":", lbp_image)
        hist = cv2.calcHist([lbp_image], [0], binary_image[idx], [256], [0, 256], hist, True).ravel()
    hist /= np.mean(hist)
    lbp_features.extend(hist)
    return lbp_features

gray_images = np.array([
[
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7]
],
[
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7]
]
], dtype=np.uint8)

binary_images = np.array([
[
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0]
],
[
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0]
]
], dtype=np.uint8)
# local_binary_pattern()
# lbp = local_binary_pattern(gray_images[0], 8, 3, method='default')
# print(lbp)
features = get_features(gray_images, binary_images, radius=3)
print(features)
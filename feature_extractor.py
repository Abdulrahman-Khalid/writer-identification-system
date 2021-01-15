import numpy as np
import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern

# 20 examples
# no_points = 8*3 and radius = 3 and method = 'uniform' ----> accuracy = 90%
# no_points = 8 and radius = 3 and method = 'uniform' ----> Accuracy: 85.0%, Average time: 12.17s
# no_points = 8 and radius = 3 and method = 'uniform' ----> Accuracy: 90.0%, Average time: 9.64s
# no_points = 8 and radius = 3 and method = 'default' ----> Accuracy: 100.0%, Average time: 9.73s
# no_points = 4 and radius = 3 and method = 'default' ----> Accuracy: 100.0%, Average time: 8.34s
# no_points = 2 and radius = 3 and method = 'default' ----> Accuracy: 80.0%, Average time: 4.50s
# no_points = 1 and radius = 3 and method = 'default' ----> Accuracy: 90.0%, Average time: 5.45s
# no_points = 4 and radius = 3 and method = 'ror' ----> Accuracy: 90.0%, Average time: 7.78s
# 100 examples
# no_points = 4 and radius = 3 and method = 'default' ----> Accuracy: 98.0%, Average time: 6.56s
# no_points = 4 and radius = 3 and method = 'default' with not inverted binary image ----> Accuracy: 70.0%, Average time: 5.14s
def get_features(gray_lines, binary_lines, radius=3, no_points=8,
                 method='default', verbose=False):
    features = np.zeros(256)

    for idx in range(len(gray_lines)):
        gray_line = gray_lines[idx]
        binary_line = binary_lines[idx]
        lbp = local_binary_pattern(gray_line, no_points, radius, method=method).astype(np.uint8)
        # if verbose:
        #     print(lbp)
        features = cv2.calcHist([lbp], [0], binary_line, [256], [0, 256], features, True).ravel()

    features /= np.mean(features)

    return features


def is_bigger_than_center(gray_image, center, x, y):
    if x < 0 or y < 0 or x >= gray_image.shape[0] or y >= gray_image.shape[1]:
        return 0
    return 1 if gray_image[x][y] >= center else 0


# def is_bigger_than_center(gray_image, center, x, y):
#     try:
#         if gray_image[x][y] >= center:
#             return 1
#         return 0
#     except:
#         return 0

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
    pattern = 0
    center = gray_image[x][y]
    pattern |= is_bigger_than_center(gray_image, center, x - radius, y + radius) * power_of_2[0]  # top right
    pattern |= is_bigger_than_center(gray_image, center, x, y + radius) * power_of_2[1]  # right
    pattern |= is_bigger_than_center(gray_image, center, x + radius, y + radius) * power_of_2[2]  # bottom_right
    pattern |= is_bigger_than_center(gray_image, center, x + radius, y) * power_of_2[3]  # bottom
    pattern |= is_bigger_than_center(gray_image, center, x + radius, y - radius) * power_of_2[4]  # bottom_left
    pattern |= is_bigger_than_center(gray_image, center, x, y - radius) * power_of_2[5]  # left
    pattern |= is_bigger_than_center(gray_image, center, x - radius, y - radius) * power_of_2[6]  # top left
    pattern |= is_bigger_than_center(gray_image, center, x - radius, y) * power_of_2[7]  # top

    return pattern


def lbp_features(gray_images, binary_images, radius=3, verbose=False):
    lbp_features = np.zeros(256)
    power_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]

    for idx in range(len(binary_images)):
        lbp_image = np.zeros(gray_images[idx].shape, np.uint8)
        for i in range(gray_images[idx].shape[0]):
            for j in range(gray_images[idx].shape[1]):
                lbp_image[i, j] = lbp_pixel(gray_images[idx], i, j, radius, power_of_2)
        if verbose: print(idx, ":", lbp_image)
        lbp_features = cv2.calcHist([lbp_image], [0], binary_images[idx], [256], [0, 256], lbp_features, True).ravel()
    lbp_features /= np.mean(lbp_features)
    lbp_features.extend(lbp_features)
    return lbp_features


# if __name__ == '__main__':
#     gray_images = np.array([
#         [
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7]
#         ],
#         [
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7],
#             [1, 2, 3, 4, 5, 6, 7]
#         ]
#     ], dtype=np.uint8)

#     binary_images = np.array([
#         [
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0]
#         ],
#         [
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0]
#         ]
#     ], dtype=np.uint8)
#     # local_binary_pattern()
#     # lbp = local_binary_pattern(gray_images[0], 8, 3, method='default')
#     # print(lbp)
#     features = lbp_features(gray_images, binary_images, radius=3, verbose=True)
#     print(features)

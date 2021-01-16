import argparse
import os
from time import time
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from tqdm import tqdm
from sklearn import svm
from skimage.feature import local_binary_pattern
cimport numpy as np

#######################################################################################################################

cpdef np.ndarray image_classification(np.ndarray train_images_features, list train_images_labels, np.ndarray test_image_features):
    cdef object clsfr = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
    clsfr.fit(train_images_features, train_images_labels)
    return clsfr.predict(test_image_features)

#######################################################################################################################

cpdef tuple image_preprocessing(np.ndarray gray_image):
    # Apply gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Remove document left and right margins
    gray_image = gray_image[:, 100:-30]

    # Binarization using threshold otsu
    cdef np.ndarray binary_image
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply edge detection method on the image
    cdef np.ndarray edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

    # Detect horizontal lines
    cdef np.ndarray lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=300,
                            lines=np.array([]), minLineLength=80, maxLineGap=1)

    cdef list y_values = []
    for i in range(lines.shape[0]):
        y_values.append(lines[i][0][1])

    # Initialize upper and lower lines with heuristic values in case of hough failure
    cdef int y_lowerline = 2800
    cdef int y_upperline = 650

    # Detect upper and lower lines that contains the handwritten text
    y_values.sort()

    # Detect upper and lower lines that contains the handwritten text
    cdef int document_median = binary_image.shape[1] // 2

    for idx, value in enumerate(y_values):
        if value > document_median:
            y_lowerline = value
            y_upperline = y_values[idx - 1] if idx != 0 \
                and y_values[idx - 1] < document_median else y_upperline
            break

    gray_image = gray_image[y_upperline:y_lowerline-3, :]
    binary_image = binary_image[y_upperline:y_lowerline-3, :]

    return binary_image, gray_image
#######################################################################################################################

# cpdef line_segmentation_gen(np.ndarray binary_image, np.ndarray gray_image):
#     '''
#     generator version
#     '''
#     # Apply Dilation to mix all line words together
#     cdef np.ndarray dilation_kernel = np.ones((1, 190), np.uint8)
#     cdef np.ndarray image_dilation = cv2.dilate(
#         binary_image, dilation_kernel, iterations=1).astype(np.uint8)
#     # Remove thin vertical lines to distinct overlaped lines
#     cdef np.ndarray vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (370, 1))
#     cdef np.ndarray remove_vertical = cv2.morphologyEx(
#         image_dilation, cv2.MORPH_OPEN, vertical_kernel)
#     # Find image contours which indicate lines
#     cdef list lines_contours
#     lines_contours, _ = cv2.findContours(
#         remove_vertical.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Extract lines
#     for line in lines_contours:
#         x, y, w, h = cv2.boundingRect(line)
#         if h > 30:
#             binary_line = binary_image[
#                 max(y-30, 0):min(y+h+60, binary_image.shape[1]),
#                 x:x+w
#             ]
#             gray_line = gray_image[
#                 max(y-30, 0):min(y+h+60, gray_image.shape[1]),
#                 x:x+w
#             ]
#             yield binary_line, gray_line

#######################################################################################################################


cpdef np.ndarray get_features(np.ndarray binary_image, np.ndarray gray_image):
    cdef np.ndarray features = np.zeros(256)

    '''
    generator version
    '''
    # Apply Dilation to mix all line words together
    cdef np.ndarray dilation_kernel = np.ones((1, 190), np.uint8)
    cdef np.ndarray image_dilation = cv2.dilate(
        binary_image, dilation_kernel, iterations=1).astype(np.uint8)
    # Remove thin vertical lines to distinct overlaped lines
    cdef np.ndarray vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (370, 1))
    cdef np.ndarray remove_vertical = cv2.morphologyEx(
        image_dilation, cv2.MORPH_OPEN, vertical_kernel)
    # Find image contours which indicate lines
    cdef list lines_contours
    lines_contours, _ = cv2.findContours(
        remove_vertical.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cdef list gray_lines = []
    cdef list binary_lines = []
    # Extract lines
    for line in lines_contours:
        x, y, w, h = cv2.boundingRect(line)
        if h > 30:
            binary_line = binary_image[
                max(y-30, 0):min(y+h+60, binary_image.shape[1]),
                x:x+w
            ]
            gray_line = gray_image[
                max(y-30, 0):min(y+h+60, gray_image.shape[1]),
                x:x+w
            ]
            gray_lines.append(gray_line)
            binary_lines.append(binary_line)

    for i in range(len(gray_lines)):
        lbp = local_binary_pattern(
            gray_lines[i], 8, 3, method='default'
        ).astype(np.uint8)
        features = cv2.calcHist([lbp], [0], binary_lines[i], [256],
                                [0, 256], features, True).ravel()

    return features / np.mean(features)

#######################################################################################################################

cpdef np.ndarray lbp_pipeline(np.ndarray gray_image):
    return get_features(*image_preprocessing(gray_image))

cpdef void all_features(list all_imgs, int jobs, np.ndarray out):
    for i in range(len(all_imgs)):
        out[i] = lbp_pipeline(all_imgs[i])

cpdef np.ndarray get_predictions(list all_imgs, list train_images_labels, int jobs, np.ndarray features):
    all_features(all_imgs, jobs, features)

    return image_classification(
        features[:-1],
        train_images_labels,
        np.array([features[-1]])
    )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', help='path to data dir', default='../data')
#     parser.add_argument('--results',
#                         help='path to results output file', default='../results.txt')
#     parser.add_argument('--time',
#                         help='path to time output file', default='../time.txt')
#     parser.add_argument('-j', '--jobs', default=-1, type=int,
#                         help='number of parallel jobs, -1 for maximum')
#     args = parser.parse_args()

#     test_cases = sorted_subdirectories(args.data)

#     # clear files
#     open(args.results, "w")
#     open(args.time, "w")

#     for test_case in tqdm(test_cases, desc='Test Cases', unit='case'):
#         path = os.path.join(args.data, test_case)
#         test_image_path, train_images_paths, \
#             train_images_labels = read_test_case_images(path)

#         # read all imgs before the timer
#         all_imgs = [
#             cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             for image_path in [*train_images_paths, test_image_path]
#         ]

#         # allocate buffer ahead
#         features = np.zeros((len(all_imgs), 256))

#         # ------ start timer ------ #
#         time_before = time()

#         predictions = get_predictions(all_imgs, train_images_labels, args.jobs, features)

#         test_time = time() - time_before
#         # ------ end timer ------ #

#         with open(args.results, "a") as f:
#             f.write('{}\n'.format(int(predictions[0])))

#         with open(args.time, 'a') as f:
#             f.write('{}\n'.format(test_time))

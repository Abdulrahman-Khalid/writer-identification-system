import cv2
import numpy as np
from time import time
from image_preprocessing import image_preprocessing
from image_segmentation import image_segmentation
from image_classification import image_classification
from utils import sorted_subdirectories, read_test_case_images

TESTS_PATH = "data"

test_cases = sorted_subdirectories(TESTS_PATH)
results_file = open("results.txt", "w")
times_file = open("time.txt", "w")

for test_case in test_cases:
    test_image_path, train_images_paths, train_images_labels = read_test_case_images(test_case)
    train_images_features = np.array([])
    test_image_features = np.array([]) 
    time_before = time()
    for image_path in np.append(train_images_paths, test_image_path):
        gray_image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        threshold, binary_image, gray_image = image_preprocessing(gray_image)
        line_contours = line_segmentation(binary_image, threshold)
        feature_vector = get_features(gray_image, binary_image, line_contours, radius=3, no_points= 3*8, method='uniform'):
        if image_path == test_image_path:
            test_image_features.append(feature_vector)
        else:
            train_images_features.append(feature_vector)
    prediction = image_classification(train_images_features, train_images_labels, test_image_features)
    test_time = time() - time_before
    results_file.write(prediction)
    times_file.write(str(test_time))
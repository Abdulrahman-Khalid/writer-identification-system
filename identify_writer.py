import cv2
import numpy as np
from time import time
from image_preprocessing import image_preprocessing
from image_segmentation import line_segmentation
from image_classification import image_classification
from feature_extractor import get_features
from utils import sorted_subdirectories, read_test_case_images

TESTS_PATH = "data"

test_cases = sorted_subdirectories(TESTS_PATH)
results_file = open("results.txt", "w")
times_file = open("time.txt", "w")

for test_case in test_cases:
    test_image_path, train_images_paths, train_images_labels = read_test_case_images(test_case)
    all_paths = np.append(train_images_paths, test_image_path)

    # read all imgs before the timer
    all_imgs = [
        (cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), image_path == test_image_path)
        for image_path in all_paths
    ]

    # faster to use list.append, then np.array it
    # than using np.append with numpy arrays
    train_images_features = []
    test_image_features = []

    time_before = time()
    for gray_image, is_test_img in all_imgs:
        binary_image, gray_image = image_preprocessing(gray_image)
        line_contours = line_segmentation(binary_image)
        feature_vector = get_features(gray_image, binary_image, line_contours)

        if is_test_img:
            test_image_features.append(feature_vector)
        else:
            train_images_features.append(feature_vector)

    predictions = image_classification(
        np.array(train_images_features), 
        train_images_labels, 
        np.array(test_image_features)
    )
    test_time = time() - time_before

    results_file.write(f'{int(predictions[0])}\n')
    times_file.write(f'{test_time:.2f}\n')

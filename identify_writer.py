import os
import cv2
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from feature_extractor import get_features
from image_segmentation import line_segmentation
from image_preprocessing import image_preprocessing
from image_classification import image_classification
from utils import sorted_subdirectories, read_test_case_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='enable verbose logging', action='store_true')
    parser.add_argument('--data', help='path to data dir', default='data')
    parser.add_argument('--results', help='path to results output file', default='results.txt')
    parser.add_argument('--time', help='path to time output file', default='time.txt')
    args = parser.parse_args()

    test_cases = sorted_subdirectories(args.data)

    # clear files
    open(args.results, "w")
    open(args.time, "w")

    for test_case in tqdm(test_cases, desc='Test Cases', unit='case'):
        path = os.path.join(args.data, test_case)
        test_image_path, train_images_paths, train_images_labels = read_test_case_images(path)
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
            lines_boxes = line_segmentation(binary_image)
            feature_vector = get_features(gray_image, binary_image, lines_boxes, verbose=args.verbose)

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

        with open(args.results, "a") as f:
            f.write(f'{int(predictions[0])}\n')

        with open(args.time, 'a') as f:
            f.write(f'{test_time:.2f}\n')

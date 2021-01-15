import os
import cv2
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from feature_extractor import get_features, lbp_features
from image_segmentation import line_segmentation
from image_preprocessing import image_preprocessing
from image_classification import image_classification


def sorted_subdirectories(path):
    return sorted([f.name for f in os.scandir(path) if f.is_dir()], key=lambda x: int(x))


def directory_files(path):
    return [f.path for f in os.scandir(path) if f.is_file()]

def read_test_case_images(path):
    train_images_paths = []
    train_images_labels = []
    test_image_path = directory_files(path)[0]
    for root, writers, _ in os.walk(path, topdown=False):
        for writer in writers:
            for image in os.scandir(os.path.join(root, writer)):
                train_images_paths.append(image.path)
                train_images_labels.append(int(writer))
    return test_image_path, train_images_paths, train_images_labels

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
        (image_path, image_path == test_image_path)
        for image_path in all_paths
    ]

    # faster to use list.append, then np.array it
    # than using np.append with numpy arrays
    train_images_features = []
    test_image_features = []
    train_labels = []

    time_before = time()
    for idx, (image_path, is_test_img) in enumerate(all_imgs):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        binary_image, gray_image = image_preprocessing(gray_image)
        binary_lines, gray_lines = line_segmentation(binary_image, gray_image)

        if is_test_img:
            feature_vector = get_features(gray_lines, binary_lines, verbose=args.verbose)
            # feature_vector = lbp_features(gray_lines, binary_lines)
            test_image_features.append(feature_vector)
        else:
            feature_vector = get_features(gray_lines, binary_lines, verbose=args.verbose)
            # feature_vector = lbp_features(gray_lines, binary_lines)
            train_images_features.append(feature_vector)

        """
        for line_contour in line_contours:
            cv2.rectangle(binary_image, (line_contour[2], line_contour[0]),(line_contour[3], line_contour[1]), 0, 1)
        cv2.imwrite("preprocessed_images/" + image_path, binary_image)
        """
    predictions = image_classification(
        np.array(train_images_features), 
        train_images_labels, 
        np.array(test_image_features).reshape(1, -1)
    )
    test_time = time() - time_before

    with open(args.results, "a") as f:
        f.write(f'{int(predictions)}\n')
    
    with open(args.time, 'a') as f:
        f.write(f'{test_time:.2f}\n')
"""
import math
import cv2
import numpy as np
import os
from image_preprocessing import image_preprocessing
from image_segmentation import line_segmentation

def directory_files(path):
    return [f.name for f in os.scandir(path) if f.is_file()]

for image_name in directory_files("test_images/"):
    path = "/".join(("test_images/", image_name))
    image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    binary_image, gray_image = image_preprocessing(image)
    line_contours = line_segmentation(binary_image)
    for line_contour in line_contours:
        x, y, w, h = cv2.boundingRect(line_contour)
        if(h > 30):
            image = cv2.rectangle(binary_image, (x, y - 30 ),( x + w, y + h + 60), 0, 1)
    path = "/".join(("preprocessed_images/", image_name))
    cv2.imwrite(path, binary_image)
"""

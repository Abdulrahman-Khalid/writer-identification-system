import argparse
import os
from time import time

import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

from feature_extractor import get_features
from image_classification import image_classification
from image_preprocessing import image_preprocessing
from image_segmentation import line_segmentation


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


def resize(img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def hog_pipeline(gray_image, **kwargs):
    if 'binary_image' in kwargs:
        binary_image = kwargs['binary_image']
    else:
        binary_image, _ = image_preprocessing(gray_image)

    return hog(resize(binary_image, 40), feature_vector=True, block_norm='L2-Hys')[:1000]


def hu_moments_pipeline(gray_image, **kwargs):
    binary_image, gray_image = image_preprocessing(gray_image)
    binary_lines, _ = line_segmentation(binary_image, gray_image)
    hus = []
    for line in binary_lines:
        hus.extend(cv2.HuMoments(cv2.moments(line)).flatten())
    return hus[:7*4]


def hu_moments_window_pipeline(gray_image, **kwargs):
    if 'binary_image' in kwargs:
        binary_image = kwargs['binary_image']
    else:
        binary_image, _ = image_preprocessing(gray_image)
    size = kwargs.get('size', 70000)
    wsize = 13

    hus = []
    x, y = binary_image.shape
    for i in range(0, x, wsize):
        for j in range(0, y, wsize):
            window = binary_image[i:i+wsize, j:j+wsize]
            hus.extend(cv2.HuMoments(cv2.moments(window)).flatten())
            if len(hus) >= size:
                return hus[:size]
    return hus[:size]


def hu_hog(gray_image, **kwargs):
    binary_image, _ = image_preprocessing(gray_image)
    a = list(hu_moments_pipeline(gray_image))
    b = list(hog_pipeline(gray_image, binary_image=binary_image))
    return a + b


def huw_hog(gray_image, **kwargs):
    binary_image, _ = image_preprocessing(gray_image)
    a = list(hu_moments_window_pipeline(
        gray_image,
        binary_image=binary_image,
        size=1000
    ))
    b = list(hog_pipeline(gray_image, binary_image=binary_image))
    return a + b


def lbp_pipeline(gray_image, **kwargs):
    binary_image, gray_image = image_preprocessing(gray_image)
    binary_lines, gray_lines = line_segmentation(binary_image, gray_image)
    return get_features(gray_lines, binary_lines)


def all_features_opt(all_imgs, out):
    for i in range(len(all_imgs)):
        out[i] = lbp_pipeline(all_imgs[i])


def all_features(all_imgs, pipeline):
    return np.array([pipeline(img) for img in all_imgs])


pipelines = {
    'lbp': lbp_pipeline,
    'hog': hog_pipeline,
    'hu': hu_moments_pipeline,
    'huw': hu_moments_window_pipeline,
    'hu+hog': hu_hog,
    'huw+hog': huw_hog,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data dir', default='data')
    parser.add_argument('--results',
                        help='path to results output file', default='results.txt')
    parser.add_argument('--time',
                        help='path to time output file', default='time.txt')
    parser.add_argument('--pipeline', default='lbp',
                        choices=list(pipelines.keys()))
    args = parser.parse_args()

    test_cases = sorted_subdirectories(args.data)

    # clear files
    open(args.results, "w")
    open(args.time, "w")

    for test_case in tqdm(test_cases, desc='Test Cases', unit='case'):
        path = os.path.join(args.data, test_case)
        test_image_path, train_images_paths, \
            train_images_labels = read_test_case_images(path)

        # read all imgs before the timer
        all_imgs = [
            cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for image_path in [*train_images_paths, test_image_path]
        ]

        # allocate buffer ahead
        features = np.zeros((len(all_imgs), 256))

        # ------ start timer ------ #
        time_before = time()

        if args.pipeline == 'lbp':
            all_features_opt(all_imgs, features)
        else:
            features = all_features(all_imgs, pipelines[args.pipeline])

        train_images_features = features[:-1]
        test_image_features = np.array([features[-1]])
        predictions = image_classification(
            train_images_features,
            train_images_labels,
            test_image_features,
        )

        test_time = time() - time_before
        # ------ end timer ------ #

        with open(args.results, "a") as f:
            f.write(f'{int(predictions[0])}\n')

        with open(args.time, 'a') as f:
            f.write(f'{test_time:.2f}\n')

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
from skimage.feature import hog


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
            if len(hus) >= size: return hus[:size]
    return hus[:size]

def hu_hog(gray_image, **kwargs):
    binary_image, _ = image_preprocessing(gray_image)
    a = list(hu_moments_pipeline(gray_image))
    b = list(hog_pipeline(gray_image, binary_image=binary_image))
    return a + b

def huw_hog(gray_image, **kwargs):
    binary_image, _ = image_preprocessing(gray_image)
    a = list(hu_moments_window_pipeline(gray_image, binary_image=binary_image, size=1000))
    b = list(hog_pipeline(gray_image, binary_image=binary_image))
    return a + b

def lbp_pipeline(gray_image, **kwargs):
    binary_image, gray_image = image_preprocessing(gray_image)
    binary_lines, gray_lines = line_segmentation(binary_image, gray_image)
    # feature_vector = lbp_features(gray_lines, binary_lines)
    return get_features(gray_lines, binary_lines, verbose=kwargs['verbose'])


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
    parser.add_argument('-v', '--verbose', help='enable verbose logging', action='store_true')
    parser.add_argument('--data', help='path to data dir', default='data')
    parser.add_argument('--results', help='path to results output file', default='results.txt')
    parser.add_argument('--time', help='path to time output file', default='time.txt')
    parser.add_argument('--pipeline', default='lbp', choices=list(pipelines.keys()))
    args = parser.parse_args()

    test_cases = sorted_subdirectories(args.data)

    # clear files
    open(args.results, "w")
    open(args.time, "w")

    pipeline = pipelines[args.pipeline]

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
            feature_vector = pipeline(gray_image, verbose=args.verbose)

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
            f.write('{}\n'.format(int(predictions[0])))

        with open(args.time, 'a') as f:
            f.write('{}\n'.format(test_time))

import argparse
import os
from time import time

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from tqdm import tqdm

from feature_extractor import get_features
from image_classification import image_classification, classifiers
from image_preprocessing import image_preprocessing, resize
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


def hog_pipeline(gray_image, **kwargs):
    if 'binary_image' in kwargs:
        binary_image = kwargs['binary_image']
    else:
        binary_image, _ = image_preprocessing(gray_image)

    return hog(resize(binary_image, 0.4), feature_vector=True, block_norm='L2-Hys')[:1000]


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
    return get_features(*image_preprocessing(gray_image))


def with_index(fn, i):
    def newfn(*args, **kwargs):
        return i, fn(*args, **kwargs)
    return newfn


def all_features(all_imgs, pipeline, jobs, out):
    unordered_results = Parallel(n_jobs=jobs)(
        delayed(with_index(pipeline, i))(all_imgs[i]) for i in range(len(all_imgs))
    )
    for i, arr in unordered_results:
        out[i] = arr


pipelines = {
    'lbp': lbp_pipeline,
    'hog': hog_pipeline,
    'huw': hu_moments_window_pipeline,
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
    parser.add_argument('-j', '--jobs', default=7, type=int,
                        help='number of parallel jobs, -1 for all cpus')
    parser.add_argument('--classifier',
                        default='svm-sigmoid', choices=classifiers.keys())
    args = parser.parse_args()

    test_cases = sorted_subdirectories(args.data)

    # clear files
    open(args.results, "w")
    open(args.time, "w")

    for test_case in tqdm(test_cases, desc='Test Cases', unit='case'):
        path = os.path.join(args.data, test_case)
        test_image_path, train_images_paths, \
            train_images_labels = read_test_case_images(path)

        # allocate buffer ahead
        if args.pipeline == 'lbp':
            features = np.zeros((len(train_images_paths)+1, 256))
        else:
            features = [0]*(len(train_images_paths)+1)

        # read all imgs before the timer
        all_imgs = [
            cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            for image_path in [*train_images_paths, test_image_path]
        ]

        # ------ start timer ------ #
        time_before = time()

        all_features(
            all_imgs, pipelines[args.pipeline], args.jobs, features
        )

        predictions = image_classification(
            args.classifier,
            # train
            features[:-1],
            train_images_labels,
            # test
            np.array([features[-1]]),
        )

        test_time = time() - time_before
        # ------ end timer ------ #

        with open(args.results, "a") as f:
            f.write('{}\n'.format(int(predictions[0])))

        with open(args.time, 'a') as f:
            f.write('{:.2f}\n'.format(test_time))

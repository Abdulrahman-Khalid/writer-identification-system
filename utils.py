import os
import numpy as np
TESTS_PATH = "data"


def sorted_subdirectories(path):
    return sorted([f.name for f in os.scandir(path) if f.is_dir()], key=lambda x: int(x))


def directory_files(path):
    return [f.path for f in os.scandir(path) if f.is_file()]


def read_test_case_images(test_case):
    path = "/".join((TESTS_PATH, test_case))
    test_image_path = directory_files(path)[0]
    train_images_paths = np.array([])
    train_images_labels = np.array([])
    for root, writers, _ in os.walk(path, topdown=False):
        for writer in writers:
            for image in os.scandir(os.path.join(root, writer)):
                train_images_paths = np.append(train_images_paths, image.path)
                train_images_labels = np.append(
                    train_images_labels,
                    int(writer)
                )
    return test_image_path, train_images_paths, train_images_labels

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

#######################################################################################################################

_classifiers = {
    'linear': svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'),
    'rbf': svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
    'poly': svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo'),
    'sigmoid': svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo'),
}


def image_classification(train_images_features, train_images_labels, test_image_features, kernel="sigmoid"):
    clsfr = _classifiers[kernel]
    clsfr.fit(train_images_features, train_images_labels)
    return clsfr.predict(test_image_features)

#######################################################################################################################

def show_image(image, window_name='image', save_name='image.png'):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image.shape[0] // 2, image.shape[1] // 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif key == ord('s') or key == ord('S'):  # wait for 's' key to save and exit
        cv2.imwrite(save_name, image)
        cv2.destroyAllWindows()


def image_preprocessing(gray_image):
    # Apply gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Remove document left and right margins
    gray_image = gray_image[:, 100:-30]

    # Binarization using threshold otsu
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply edge detection method on the image
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

    # Detect horizontal lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=300,
                            lines=np.array([]), minLineLength=80, maxLineGap=1)

    y_values = []
    for i in range(lines.shape[0]):
        y_values.append(lines[i][0][1])

    # Initialize upper and lower lines with heuristic values in case of hough failure
    y_lowerline = 2800
    y_upperline = 650

    # Detect upper and lower lines that contains the handwritten text
    y_values.sort()

    # Detect upper and lower lines that contains the handwritten text
    document_median = binary_image.shape[1] // 2

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

def line_segmentation_gen(binary_image, gray_image):
    '''
    generator version
    '''
    # Apply Dilation to mix all line words together
    dilation_kernel = np.ones((1, 190), np.uint8)
    image_dilation = cv2.dilate(
        binary_image, dilation_kernel, iterations=1).astype(np.uint8)
    # Remove thin vertical lines to distinct overlaped lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (370, 1))
    remove_vertical = cv2.morphologyEx(
        image_dilation, cv2.MORPH_OPEN, vertical_kernel)
    # Find image contours which indicate lines
    lines_contours, _ = cv2.findContours(
        remove_vertical.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            yield binary_line, gray_line

#######################################################################################################################


def get_features(binary_image, gray_image, radius=3, no_points=8,
                 method='default'):
    features = np.zeros(256)

    for binary_line, gray_line in line_segmentation_gen(binary_image, gray_image):
        lbp = local_binary_pattern(
            gray_line, no_points, radius, method=method
        ).astype(np.uint8)
        features = cv2.calcHist([lbp], [0], binary_line, [256],
                                [0, 256], features, True).ravel()

    return features / np.mean(features)

#######################################################################################################################
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


def lbp_pipeline(gray_image, **kwargs):
    return get_features(*image_preprocessing(gray_image))


def with_index(fn, i):
    def newfn(*args, **kwargs):
        return i, fn(*args, **kwargs)
    return newfn

def all_features(all_imgs, pipeline, jobs, out):
    n_jobs = jobs if jobs != -1 else len(all_imgs)
    unordered_results = Parallel(n_jobs=n_jobs)(
        delayed(with_index(pipeline, i))(all_imgs[i]) for i in range(len(all_imgs))
    )
    for i, arr in unordered_results:
        out[i] = arr

def get_predictions(all_imgs, train_images_labels, jobs, features):
    all_features(
        all_imgs, lbp_pipeline, jobs, features
    )

    predictions = image_classification(
        # train
        features[:-1],
        train_images_labels,
        # test
        np.array([features[-1]])
    )
    
    return predictions

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

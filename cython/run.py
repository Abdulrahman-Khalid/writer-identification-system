from identify_writer import *
import argparse
import os
from time import time
from tqdm import tqdm
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to data dir', default='../data')
    parser.add_argument('--results',
                        help='path to results output file', default='../results.txt')
    parser.add_argument('--time',
                        help='path to time output file', default='../time.txt')
    parser.add_argument('-j', '--jobs', default=-1, type=int,
                        help='number of parallel jobs, -1 for maximum')
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

        predictions = get_predictions(all_imgs, train_images_labels, args.jobs, features)

        test_time = time() - time_before
        # ------ end timer ------ #

        with open(args.results, "a") as f:
            f.write('{}\n'.format(int(predictions[0])))

        with open(args.time, 'a') as f:
            f.write('{}\n'.format(test_time))

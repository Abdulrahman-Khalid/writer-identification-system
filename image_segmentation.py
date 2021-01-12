import cv2
import numpy as np


def image_segmentation(img, threshold):
    # Method One
    # kernel = np.ones((3,50), np.uint8)
    # image_dilation = cv2.dilate(np.invert(image), kernel, iterations=1).astype(np.uint8)
    # words_contours,_ = cv2.findContours(image_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    
    # Method Two
    img_filtered = cv2.bilateralFilter(img, 9, 75, 75)
    _, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (words_contours, _) = cv2.findContours(np.invert(img_threshold), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return words_contours

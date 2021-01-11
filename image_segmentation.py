import cv2
import numpy as np

def image_segmentation(image, threshold):
    kernel = np.ones((3,50), np.uint8)
    image_dilation = cv2.dilate(np.invert(image), kernel, iterations=1).astype(np.uint8)
    words_contours,_ = cv2.findContours(image_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    return words_contours

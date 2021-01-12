import cv2


def image_preprocessing(image):
    image = image[680:2800, 150:]
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

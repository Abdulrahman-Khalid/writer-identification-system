import cv2


def image_preprocessing(gray_image):
    # Crop the image to get the handwritten text area
    gray_image = gray_image[680:2800, 150:]
    # Binarization using threshold otsu
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image, gray_image

import cv2

def image_preprocessing(image):
    # Crop the image to get the handwritten text area
    image = image[680:2800,150:]
    # Binarization using threshold otsu
    return cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
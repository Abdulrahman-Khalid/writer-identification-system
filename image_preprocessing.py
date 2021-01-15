import cv2
import numpy as np
import statistics

def image_preprocessing(gray_image):
    # Remove document left and right margins 
    y, x = gray_image.shape
    gray_image = gray_image[:,100: x-30]
     
    # Apply edge detection method on the image 
    edges = cv2.Canny(gray_image, 50, 150,apertureSize = 3) 
    
    # Detect horizontal lines
    lines = cv2.HoughLinesP(edges,rho=1,theta=np.pi/180, threshold=300,lines=np.array([]), minLineLength=80,maxLineGap=1)
    
    y_values = []
    for i in range(lines.shape[0]):
        y_values.append(lines[i][0][1])
    
    # Detect upper and lower lines that contains the handwritten text
    y_values.sort()
    document_median = gray_image.shape[1] // 2
    for idx, value in enumerate(y_values):
        if value > document_median:
            y_lowerline = value
            y_upperline = y_values[idx - 1]
            break

    gray_image = gray_image[y_upperline + 3: y_lowerline - 3,:]

    # Binarization using threshold otsu
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    return binary_image, gray_image

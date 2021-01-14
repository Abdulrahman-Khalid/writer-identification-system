import cv2
import numpy as np


def line_segmentation(img, threshold):
    # Apply Dilation to mix all line words or all words characters together
    kernel = np.ones((3,180), np.uint8)
    image_dilation = cv2.dilate(np.invert(img), kernel, iterations=1).astype(np.uint8)
    lines_contours,_ = cv2.findContours(image_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    return lines_contours


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm


def word_segmentation(img, threshold):
    kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	_, imgThres = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    words_contours, _ = cv2.findContours(np.invert(imgThres), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return words_contours


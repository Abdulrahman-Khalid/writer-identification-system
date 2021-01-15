import cv2
import numpy as np


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
    cv2.GaussianBlur(gray_image, (5, 5), 0, dst=gray_image)

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

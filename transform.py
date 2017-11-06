import cv2
import numpy as np

#
# Compute pixel gradient norms and directions, and threshold them into a binary image.
# Input image should be in HLS color space.
#
def gradient_threshold(img, norm_threshold=(0.0, 1.0), theta_threshold=(0.0, np.pi/2)):
    gray = img[:,:,1]

    # TODO play with kernel size.
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    norm = np.sqrt(dx**2 + dy**2)
    norm = norm / np.max(norm)

    theta = np.arctan2(np.absolute(dy), np.absolute(dx))

    norm_bin = np.zeros_like(gray, np.uint8)
    norm_bin[(norm >= norm_threshold[0]) & (norm <= norm_threshold[1])] = 1

    theta_bin = np.zeros_like(gray, np.uint8)
    theta_bin[(theta >= theta_threshold[0]) & (theta <= theta_threshold[1])] = 1

    return (norm_bin & theta_bin) * 255
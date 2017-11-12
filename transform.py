import cv2
import numpy as np

#
# Compute pixel gradient norms and directions, and threshold them into a binary image.
# Input image should be in HLS color space.
#
def gradient_threshold(img, norm_threshold=(0.0, 1.0), theta_threshold=(0.0, np.pi/2)):
    gray = img[...,1]

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)

    norm = np.sqrt(dx**2 + dy**2)
    norm = norm / np.max(norm)

    theta = np.arctan2(np.absolute(dy), np.absolute(dx))

    norm_bin = np.zeros_like(gray, np.uint8)
    norm_bin[(norm >= norm_threshold[0]) & (norm <= norm_threshold[1])] = 255

    theta_bin = np.zeros_like(gray, np.uint8)
    theta_bin[(theta >= theta_threshold[0]) & (theta <= theta_threshold[1])] = 255

    return (norm_bin & theta_bin)

#
# Return binary image of pixels whose S value fall within threshold.
# Input image should be in HLS color space.
#
def channel_threshold(img, threshold=(0, 255)):
    S = img[...,2]

    S_bin = np.zeros_like(S, np.uint8)
    S_bin[(S >= threshold[0]) & (S <= threshold[1])] = 255

    return S_bin

#
# Perform a perspective transform to get a bird's eye view of the road.
# Credit to sbagalka on the Udacity forum for coming up with the source and destination pixels,
# these are a slightly tweaked version.
#
src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
dst = np.float32([[(350, 720), (370, 0), (960, 0), (980, 720)]])

T = cv2.getPerspectiveTransform(src, dst)
T_inv = cv2.getPerspectiveTransform(dst, src)

def warp(img):
    return cv2.warpPerspective(img, T, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def unwarp(img):
    return cv2.warpPerspective(img, T_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


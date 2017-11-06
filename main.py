import glob
import cv2
import numpy as np
from calibration import *
from transform import *

def undistort_test(camera):

    i = 0

    # Undistort a few test images.
    for file in glob.glob('./camera_cal/test_image*.jpg'):
        img = cv2.imread(file)
        img = camera.undistort(img)
        cv2.imwrite('./output_images/undistorted{}.jpg'.format(i), img)
        i += 1

def threshold_test(camera):
    img = cv2.imread('test_images/test5.jpg')
    img = camera.undistort(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    binary = gradient_threshold(img, (0.2, 1.0), (0.7, 1.3))
    cv2.imwrite('./output_images/gradient.jpg', binary)

if __name__ == '__main__':
    camera = calibrate_camera()

    undistort_test(camera)
    threshold_test(camera)
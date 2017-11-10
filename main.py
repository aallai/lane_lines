import glob
import cv2
import numpy as np
from calibration import *
from transform import *
from lanes import *

def undistort_test(camera):
    i = 0
    # Undistort a few test images.
    for file in glob.glob('./camera_cal/test_image*.jpg'):
        img = cv2.imread(file)
        img = camera.undistort(img)

        cv2.imwrite('./output_images/undistorted{}.jpg'.format(i), img)

        i += 1

def threshold_test(camera):
    i = 0
    images = []
    for file in glob.glob('test_images/test*.jpg') + glob.glob('test_images/straight_lines*.jpg'):
        img = cv2.imread(file)
        img = camera.undistort(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        grad = gradient_threshold(img, (0.2, 1.0), (0.7, 1.3))
        channel = channel_threshold(img, (150, 255))
        combined = grad | channel

        cv2.imwrite('./output_images/dstacked{}.jpg'.format(i), np.dstack((np.zeros_like(combined), grad, channel)))
        cv2.imwrite('./output_images/binary{}.jpg'.format(i), combined)
        images.append(combined)

        i += 1

    return images

def warp_test(images):
    warped = []
    i = 0
    for img in images:
        w = warp(img)
        cv2.imwrite('./output_images/warped{}.jpg'.format(i), w)
        warped.append(w)

        i += 1

    return warped

def find_lanes_test(images):
    i = 0
    for img in images:
        fit_poly(img, './output_images/lanes{}.jpg'.format(i))
        i += 1

if __name__ == '__main__':
    camera = calibrate_camera()

    undistort_test(camera)
    images = threshold_test(camera)
    warped = warp_test(images)
    find_lanes_test(warped)











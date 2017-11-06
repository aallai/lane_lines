import glob
import cv2
from calibration import *

if __name__ == '__main__':
    camera = calibrate_camera()

    i = 0

    # Undistort a few test images.
    for file in glob.glob('./camera_cal/test_image*.jpg'):
        img = cv2.imread(file)
        img = camera.undistort(img)
        cv2.imwrite('./output_images/undistorted{}.jpg'.format(i), img)
        i += 1
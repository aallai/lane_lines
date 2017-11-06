import glob
import cv2
import numpy as np

NUM_CORNERS_X = 9
NUM_CORNERS_Y = 6

class Camera():
    def __init__(self, M, dist):
        self.camera_matrix = M
        self.distortion_coeffs = dist

    def undistort(self, img):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)


# Calibrate camera based on chessboard images.
def calibrate_camera():
    img_points = []
    obj_points = []
    image_shape = None

    for file in glob.glob('./camera_cal/calibration*.jpg'):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape

        ret, corners = cv2.findChessboardCorners(gray, (NUM_CORNERS_X, NUM_CORNERS_Y))

        if ret == 0:
            raise Exception('Failed to find chessboard pattern in calibration image {}.'.format(file))

        img_points.append(corners)

        points = np.zeros((NUM_CORNERS_X * NUM_CORNERS_Y, 3), np.float32)
        points[:,:2] = np.mgrid[0:NUM_CORNERS_X, 0:NUM_CORNERS_Y].T.reshape(-1,2)
        obj_points.append(points)

    err, camera_matrix, distortion_coeffs, _, __ = cv2.calibrateCamera(obj_points, img_points, image_shape, None, None)
    return Camera(camera_matrix, distortion_coeffs)
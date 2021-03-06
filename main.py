import argparse
import glob
import cv2
import numpy as np
from calibration import *
from transform import *
from lanes import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def undistort(camera, images):
    undistorted = []
    for img in images:
        img = camera.undistort(img)
        undistorted.append(img)

    return undistorted

def bgr_to_hls(images):
    hls = []

    for img in images:
        hls.append(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))

    return hls

def threshold(images, dstack=False):
    dstacked = []
    thresholded = []

    for img in images:

        grad = gradient_threshold(img, (0.2, 1.0), (0.7, 1.3))
        channel = channel_threshold(img, (150, 255))
        combined = grad | channel

        thresholded.append(combined)

        if dstack:
            dstacked.append(np.dstack((np.zeros_like(combined), grad, channel)))

    return thresholded, dstacked

def perspective_transform(images):
    warped = []
    for img in images:
        w = warp(img)
        warped.append(w)

    return warped

def find_lanes(images, write=False):
    i = 0
    polys = []
    curvatures = []
    for img in images:
        fit = fit_poly(img, None, None, './output_images/lanes{}.jpg'.format(i) if write else '')
        curv = lane_curvature(fit[3], fit[4])
        polys.append(fit)
        curvatures.append(curv)
        i += 1

    return polys, curvatures

def draw_lanes(images, polys, curvatures, flip_color=False):

    if (len(images) != len(polys)):
        raise Exception('Number of images and polynomials differ!')

    with_lanes = []

    for i in range(len(images)):
        img = images[i]
        canvas = np.zeros_like(img, dtype=np.uint8)

        y, x_l, x_r, _, __ = polys[i]
        left = np.array([np.transpose(np.vstack([x_l, y]))])
        right = np.array([np.flipud(np.transpose(np.vstack([x_r, y])))])
        points = np.hstack((left, right))

        cv2.fillPoly(canvas, np.int32(points), (0,255, 0))

        canvas = unwarp(canvas)
        img = cv2.addWeighted(img, 1, canvas, 0.3, 0)

        cv2.putText(img, 'l-radius: %.2fm, r-radius: %.2fm, offset: %.2fm' % curvatures[i], (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255) if flip_color else (255,0,0))
        with_lanes.append(img)

    return with_lanes

def write_images(images, prefix):
    i = 0
    for img in images:
        cv2.imwrite('./output_images/' + prefix + '{}'.format(i) + '.jpg', img)
        i += 1

def test():
    camera = calibrate_camera()

    images = []
    for file in glob.glob('test_images/test*.jpg') + glob.glob('test_images/straight_lines*.jpg'):
        images.append(cv2.imread(file))

    undistorted = undistort(camera, images)
    write_images(undistorted, 'undistorted')

    hls = bgr_to_hls(undistorted)
    binary, dstacked = threshold(hls, True)

    write_images(binary, 'binary')
    write_images(dstacked, 'dstack')

    warped = perspective_transform(binary)
    write_images(warped, 'warped')

    polys, curvatures = find_lanes(warped, True)

    final_images = draw_lanes(images, polys, curvatures, True)

    write_images(final_images, 'final')

def test_calibration():
    camera = calibrate_camera()
    img = cv2.imread('./camera_cal/test_image.jpg')
    cv2.imwrite('./output_images/chessboard.jpg', camera.undistort(img))

# Smoothing factor.
λ = 0.3

prev_l = []
prev_r = []

def process_image(img):

    global prev_l
    global prev_r

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    fit = fit_poly(perspective_transform(threshold([hls])[0])[0], None if len(prev_l) == 0 else prev_l[0], None if len(prev_r) == 0 else prev_r[0])
    y, x_l, x_r, left_poly, right_poly = fit
    curvature = lane_curvature(left_poly, right_poly)
    l_curv, r_curv, offset = curvature

    # If the lane is an outlier, use previous lanes for drawing.
    if is_outlier(x_l[0], x_r[0], l_curv, r_curv, offset):
        if (len(prev_l) > 0) and len(prev_r) > 0:
            return draw_lanes([img], [(y, prev_l, prev_r, left_poly, right_poly)], [curvature])[0]
        else:
            return img

    if len(prev_l) > 0 and len(prev_r) > 0:
        x_l = np.int32(λ * np.float64(x_l) + (1 - λ) * np.float64(prev_l))
        x_r = np.int32(λ * np.float64(x_r) + (1 - λ) * np.float64(prev_r))

    prev_l = x_l
    prev_r = x_r
    return draw_lanes([img], [(y, x_l, x_r, left_poly, right_poly)], [curvature])[0]

def video(video, out_file):

    camera = calibrate_camera()

    process = lambda img : process_image(camera.undistort(img))

    video = VideoFileClip(video)
    output = video.fl_image(process)
    output.write_videofile(out_file, audio=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane Finding')
    parser.add_argument('-c', '--calibration', action="store_true", default='', help='Undistort chessboard image.')
    parser.add_argument('-t', '--test', action="store_true", default='', help='Test algorithms on images in test_images directory.')
    parser.add_argument('-v', '--video', type=str, default='', help='Run video pipeline on specified video clip.')
    parser.add_argument('-o', '--output', type=str, default='output_video.mp4', help='Name of processed video file.')
    args = parser.parse_args()

    if args.calibration:
        test_calibration()
    elif args.test:
        test()
    elif args.video:
        video(args.video, args.output)
    else:
        parser.print_help()











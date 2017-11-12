import cv2
import numpy as np
import matplotlib.pyplot as plt

LANE_LENGTH = 30
LANE_WIDTH = 3.7

LANE_LENGTH_PIXELS = 720
LANE_WIDTH_PIXELS = 620
LANE_EPSILON = 50
OFFSET_THRESHOLD = 1.0
CURVATURE_RADIUS_THRESHOLD = 200

IMAGE_SHAPE = (720, 1280, 3)

# Return img with lane visualizations overlayed.
def visualize_lanes(img, window_height, window_width, left_x, right_x, left_poly, right_poly, out_file):

    def set_pixels(img, i, center):
        img[window_height*i:window_height*(i+1), max(center-window_width//2, 0):min(center+window_width//2, img.shape[1])] = 255
        return img

    windows = np.zeros_like(img, np.uint8)
    num_points = len(left_x)

    for i in reversed(range(num_points)):
        windows = set_pixels(windows, i, left_x[num_points - i - 1])
        windows = set_pixels(windows, i, right_x[num_points - i - 1])

    canvas = np.zeros_like(img)
    windows = np.array(cv2.merge((windows, canvas, canvas)), np.uint8)
    color_img = np.array(cv2.merge((img, img, img)), np.uint8)
    out_img = cv2.addWeighted(color_img, 1, windows, 0.5, 0.0)

    y = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    px_l = np.polyval(left_poly, y)
    px_r = np.polyval(right_poly, y)

    curve_left, curve_right, offset = lane_curvature(left_poly, right_poly)

    plt.imshow(out_img)
    plt.plot(px_l, y, color='yellow')
    plt.plot(px_r, y, color='yellow')
    plt.title('l-radius: %.2fm, r-radius: %.2fm, offset: %.2fm' % (curve_left, curve_right, offset))
    plt.savefig(out_file)
    plt.gcf().clear()

#
# Helper used in fit_poly below. When a patch has no pixels set, np.argmax returns
# the first element in the array. This biases the dotted lanes to curve left. Use the midpoint
# of the array instead.
#
def middle_max(l):
    m = np.argmax(l)
    maxes = np.argwhere(l == m).flatten().tolist()
    return m if len(maxes) == 0 else maxes[len(maxes)//2]

#
# Takes a binary image containing a top-down view of the lanes, and returns
# a polynomial fit for each lane. Optionally writes an image with a visualization of the fit overlayed.
#
# Implemented by convoluting a window across a horizontal slice of the image to find a patch with the most
# pixels set. This patch is considered to be where the lane is.
#
def fit_poly(img, prev_left=None, prev_right=None, out_file='', window_width=40, window_height=80, search_radius=LANE_EPSILON):

    left_x = []
    right_x = []
    window = np.ones(window_width)
    half_image = img.shape[1]//2
    full_image = img.shape[1]
    half_window = window_width//2
    ny = img.shape[0] // window_height

    for i in reversed(range(ny)):

        hist = np.sum(img[window_height*i:window_height*(i+1),...], axis=0)
        conv = np.convolve(hist, window)

        left_search_endpoint = 0
        right_search_endpoint = half_image

        if prev_left:
            left_search_endpoint = max(prev_left + half_window - search_radius, 0)
            right_search_endpoint = min(prev_left + half_window + search_radius, full_image)

        conv_max = middle_max(conv[left_search_endpoint:right_search_endpoint])
        left = min(max(conv_max + left_search_endpoint - half_window, 0), full_image)

        left_search_endpoint = half_image
        right_search_endpoint = len(conv)

        if prev_right:
            left_search_endpoint = max(prev_right + half_window - search_radius, 0)
            right_search_endpoint = min(prev_right + half_window + search_radius, full_image)

        conv_max = middle_max(conv[left_search_endpoint:right_search_endpoint])
        right = max(min(conv_max + left_search_endpoint - half_window, full_image), 0)

        left_x.append(left)
        right_x.append(right)

        prev_left = left
        prev_right = right

    y = np.linspace(img.shape[0], window_height, (img.shape[0] // window_height))

    left_poly = np.polyfit(y, left_x, 2)
    right_poly = np.polyfit(y, right_x, 2)

    if out_file != '':
        visualize_lanes(img, window_height, window_width, left_x, right_x, left_poly, right_poly, out_file)


    return y, left_x, right_x, left_poly, right_poly

# Return radius of curvature in meters.
# Maps pixels to meters using a lane size which is probably off somewhat.
def lane_curvature(left_poly, right_poly):
    m_per_y = LANE_LENGTH / LANE_LENGTH_PIXELS
    m_per_x = LANE_WIDTH / LANE_WIDTH_PIXELS

    y = np.linspace(0, IMAGE_SHAPE[0]-1, num=IMAGE_SHAPE[0])

    left_x = np.polyval(left_poly, y)
    right_x = np.polyval(right_poly, y)

    offset = (((right_x[-1] - left_x[-1])//2) - IMAGE_SHAPE[0]//2) * m_per_x

    left_poly = np.polyfit(y * m_per_y, left_x * m_per_x, 2)
    right_poly = np.polyfit(y * m_per_y, right_x * m_per_x, 2)

    curve_left = ((1 + (2 * left_poly[0] * 720 * m_per_y + left_poly[1])**2)**1.5) / np.absolute(2 * left_poly[0])
    curve_right = ((1 + (2 * right_poly[0] * 720 * m_per_y + right_poly[1])**2)**1.5) / np.absolute(2 * right_poly[0])

    return curve_left, curve_right, offset

def is_outlier(x_l, x_r, l_curv, r_curv, offset):
    if np.absolute(np.absolute(x_l - x_r) - LANE_WIDTH_PIXELS) > LANE_EPSILON:
        return True

    if np.absolute(l_curv) < CURVATURE_RADIUS_THRESHOLD or np.absolute(r_curv) < CURVATURE_RADIUS_THRESHOLD:
        return True

    if np.absolute(offset) > OFFSET_THRESHOLD:
        return True

    return False


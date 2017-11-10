import cv2
import numpy as np

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
    windows = np.array(cv2.merge((canvas, canvas, windows)), np.uint8)
    color_img = np.array(cv2.merge((img, img, img)), np.uint8)
    out_img = cv2.addWeighted(color_img, 1, windows, 0.5, 0.0)
    cv2.imwrite(out_file, out_img)

eps = 0.001

def middle_max(l):
    m = np.argmax(l)
    maxes = np.argwhere(l == m).flatten().tolist()
    return m if len(maxes) == 0 else maxes[len(maxes)//2]

#
# Takes a binary image containing a top-down view of the lanes, and returns
# a polynomial fit for each lane. Optionally writes an image with the fit overlayed.
#
def fit_poly(img, out_file='', window_width=30, window_height=30, search_radius=30):

    left_x = []
    right_x = []
    window = np.ones(window_width)
    half_image = img.shape[1]//2
    half_window = window_width//2

    # Run first convolution across entire botton of image to find inital lines.
    # Left half first.
    ny = img.shape[0] // window_height

    hist = np.sum(img[window_height*(ny-1):, :half_image], axis=0)
    conv = np.convolve(hist, window)

    # With 'full' padding, convolution indices correspond to right side of window.
    left = max(np.argmax(conv) - half_window, 0)

    hist = np.sum(img[window_height*(ny-1):, half_image:], axis=0)
    conv = np.convolve(hist, window)

    right = min(np.argmax(conv) - half_window + half_image, img.shape[1])

    left_x.append(left)
    right_x.append(right)

    for i in reversed(range(ny-1)):

        hist = np.sum(img[window_height*i:window_height*(i+1), :half_image], axis=0)
        conv = np.convolve(hist, window)

        left_search_endpoint = max(left + half_window - search_radius, 0)
        right_search_endpoint = min(left + half_window + search_radius, half_image)

        conv_max = middle_max(conv[left_search_endpoint:right_search_endpoint])
        left = max(conv_max + left_search_endpoint - half_window, 0)

        hist = np.sum(img[window_height*i:window_height*(i+1), half_image:], axis=0)
        conv = np.convolve(hist, window)

        # Need to translate right into [0, half_image].
        left_search_endpoint = max(right + half_window - search_radius - half_image , 0)
        right_search_endpoint = min(right + half_window + search_radius - half_image, half_image)

        conv_max = middle_max(conv[left_search_endpoint:right_search_endpoint])
        right = min(conv_max + left_search_endpoint - half_window + half_image, img.shape[1])

        left_x.append(left)
        right_x.append(right)

    y = np.linspace(img.shape[0], window_height, (img.shape[0] // window_height))

    left_poly = np.polyfit(y, left_x, 2)
    right_poly = np.polyfit(y, right_x, 2) 

    if out_file != '':
        visualize_lanes(img, window_height, window_width, left_x, right_x, left_poly, right_poly, out_file)


    return left_poly, right_poly

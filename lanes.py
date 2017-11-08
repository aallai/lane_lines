import cv2

#
# Takes a binary image containing a top-down view of the lanes, and returns
# a polynomial fit for each lane. Optionally writes an image with the fit overlayed.
#
def fit_poly(img, window_width=50, window_height=80, search_radius=100, out_file=''):

    left_points = []
    right_point = []
    window = np.ones(window_width)

    # With 'full' padding, convolution indices correspond to right side of window.
    to_conv = lambda x : x + window_width//2
    from_conv = lambda x : x - window_width//2

    # Run first convolution across entire botton of image to find inital lines.
    # Left half first.
    ny = img.shape[0] // window_height
    hist = np.sum(img[window_height*(ny-1):, :img.shape[1]//2], axis=0)
    conv = np.convolve(hist, window)

    left = max(from_conv(np.argmax(conv)), 0)

    hist = np.sum(img[window_height*(ny-1):, img.shape[1]//2:], axis=0)
    conv = np.convolve(hist, window)

    right = min(from_conv(np.argmax(conv)), img.shape[1])

    left_points.append(left)
    right_points.append(right)

    for i in reversed(range(ny-2)):
        hist = np.sum(img[window_height*i:window_height*(i+1), :img.shape[1]//2], axis=0)
        conv = np.convolve(hist, window)

        left_search_endpoint = max(to_conv(left) - search_radius, 0)
        right_search_endpoint = min(to_conv(left) + search_radius, img.shape[1])

        left = max(from_conv(np.argmax(conv[left_search_endpoint:right_search_endpoint])), 0)

        hist = np.sum(img[window_height*i:window_height*(i+1), img.shape[1]//2:], axis=0)
        conv = np.convolve(hist, window)

        left_search_endpoint = max(to_conv(right) - search_radius, 0)
        right_search_endpoint = min(to_conv(right) + search_radius, img.shape[1])

        right = min(from_conv(np.argmax(conv[left_search_endpoint:right_search_endpoint])), img.shape[1])

        left_points.append(left)
        right_points.append(right)

    return left_points, right_points

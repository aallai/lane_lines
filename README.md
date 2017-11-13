## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

[//]: # (Image References)

[chess1]: ./camera_cal/test_image.jpg "Distorted"
[chess2]: ./output_images/chessboard.jpg "Undistorted"
[dist]:   ./test_images/test6.jpg "Distorted"
[undist]: ./output_images/undistored6.jpg "Undistorted"
[dstacked]: ./output_images/dstacked2.jpg "Stacked (green is gradient, red is S channel)"
[binary]: ./output_images/binary4.jpg "Binary"

[driverview]: ./output_images/binary6.jpg "Driver's perspective"
[birdview]: ./output_images/warped6.jpg "Bird's eye view"

[lanesgood]: ./output_images/lanes0.jpg "Good lane detection"
[lanesbad]: ./output_images/lanes3.jpg "Failure due to noise"

[final]: ./output_images/final4.jpg "Lane surface drawing"
[video]: ./output_images/video.gif "Lane detection"

## Rubric Points

### Camera Calibration

The code for this step is contained in calibration.py file. I largely follow the steps layed out in the lectures. First I get OpenCV to find chessboard corners in the calibration images. Some of these don't have a sufficient margin around some of the corners so I use a subset of the images provided. The coordinates for these corners a passed in as the image plane coordinates. The world coordinates are passed in as a grid of evenly spaced points in x-y, all at z = 0 (so the camera is considered rotated/translated in some of the calibration images). I then get OpenCV to solve for the camera projection matrix and the distortion coefficients.

Here is an example of a distorted/undistorted chessboard:

![alt text][chess1] ![alt text][chess2]

### Pipeline (single images)

#### 1. Distortion

After calibrating the camera, all images are undistorted:

![alt text][dist] ![alt text][undist]

#### 2. Binary transforms

I use 2 channels for my binary images: 1. all pixels whose gradient magnitude and orientation fit certain thresholds, and 2. a threshold on the S channel of the images (after converting to HLS color space). The code is is transform.py with visualization in main.py. Here is an illustration with the tree-shadowed frame:

![alt text][dstacked] ![alt text][binary]

Notice there is some noise in there.

#### 3. Perspective transform.

The code for warping is in transform.py. It consists of calling the getPerspective and warpPerspective OpenCV functions with these source points:

```python
src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
dst = np.float32([[(350, 720), (370, 0), (960, 0), (980, 720)]])
```

These are a tweaked version of the points posted by sbagalka on the forum. The results look pretty good:

![alt text][driverview] ![alt text][birdview]

#### 4. Polynomial fit

I used the convolution-based algorithm to find the lane lines. The code is in lanes.py under the fit_poly function. The search starts using either the relevant half of the image, or a 1D-radius around the previous x values, if in video pipeline mode. After the first lane points are found, the image is searched using the previous x values.

I make a minor modification, which is to use the median of the maximum activation positions, if more than one window position has max activation. This because when the entire search area is black, np.argmax will return the first position, which biases the dotted lanes to curve left. By picking the middle position instead, I bias them to be straight.

Here is an example of lane detections, including a failure case.

![alt text][driverview] ![alt text][birdview]

#### 5. Radius of curvature, offset from center.

This is under the lane_curvature function in lanes.py. I followed the derivation from the lectures, and used simple math to find the offset from center. I assume the camera is center mounted. The radius of curvature is is the ~1km ballpark suggested as a sanity check.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented by draw_lanes in main.py. I use OpenCV fillPoly, as demonstrated in the lectures, and overlay some text with the curvature/offset informaton:

![alt text][final]

---

### Pipeline (video)

#### 1.

The video pipeline tracks the lane detections in the previous frame to seed the lane search in the current frame. Outlier detection is used to reject bad lane detections. Lanes where the lane width is too small or the curavture or offset are too large are dropped, and the previous lane detections are drawn instead.

Here's a [link to my video result](./output_video.mp4)

Here's a gif.

![alt text][video]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think noise in the binary images is the biggest problem (at least in the easy video). This could be remedied by zeroing out anything outside the region where we expect the lanes to be. This might work fine on the highway, but on lower speed roads with high curvature, it might end up cropping out the lanes.

I think the color thresholding has room for improvement too. Right now I just do a simple S channel threshold, and it picks up some of the shadows and other noise. I could do more research on representations of white & yellow is HLS, and make improvements there.

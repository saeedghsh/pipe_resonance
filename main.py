#!/usr/bin/env python
"""Main script to launch examples from."""

import os
import numpy
import cv2
from typing import Tuple, Optional, List, Any
import matplotlib.pyplot as plt

from user_defined_types import Pixel
from scene_configuration import SceneConfiguration
from image_morph import equalize, blur, binarize, skeletonize
from image_draw import RED, BLUE, draw_points, write, draw_scene_configuration

RESIZE = 0.2  # resizing factor of the image

EQUALIZATION_MODE = ["histogram", "clahe", None, ][1]
EQUALIZATION_KERNEL_SIZE = 8

BLURRING_MODE = ["blur", "GaussianBlur", "medianBlur", "bilateralFilter", None, ][0]
BLURRING_KERNEL_SIZE = 15

BINARIZATION_MODE = ["thresh_binary", "thresh_otsu", "adaptive_thresh_gaussian", "adaptive_thresh_mean", None, ][1]
BINARIZATION_THRESHOLD = 127
BINARIZATION_KERNEL_SIZE = 9

PIPE_WIDTH_PIXEL = 10


# TODO: this need rethinking!
def _points_from_skeleton(skeleton: numpy.ndarray) -> numpy.ndarray:
    points_row, points_col = numpy.nonzero(skeleton)

    points = numpy.array([points_row, points_col]).T

    # TODO: we need to thin out the points
    # approach 1) remove any point with less than 8 neighbors
    # compute the distance of each point to all others
    # count the number of distances <= sqrt(2)  # by increasing we can make neighborhood bigger
    from scipy.spatial.distance import pdist, squareform
    distance_matrix = squareform(pdist(points))
    neighborhood_distance = numpy.sqrt(2) + numpy.finfo(float).eps
    counts_of_neighbors = numpy.count_nonzero(distance_matrix <= neighborhood_distance, axis=0)
    number_of_expected = 9  # this includes the point itself
    selected_points_indices = numpy.nonzero(counts_of_neighbors >= number_of_expected)[0]
    return points[selected_points_indices]


def _process_frame(
    image: numpy.ndarray,
    scene_configuration: SceneConfiguration,
    resize: Optional[float] = RESIZE,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if resize is None:
        gray_resized = gray.copy()
    else:
        gray_resized = cv2.resize(gray, (0,0), fx=resize, fy=resize)

    if scene_config.is_set():
        # block out the checkerboard
        xmin = scene_configuration.checkerboard.top_left.x
        ymin = scene_configuration.checkerboard.top_left.y
        xmax = scene_configuration.checkerboard.bottom_right.x
        ymax = scene_configuration.checkerboard.bottom_right.y
        gray_resized[ymin:ymax, xmin:xmax] = 0

        # crop out the backdrop
        xmin = scene_configuration.backdrop.top_left.x
        ymin = scene_configuration.backdrop.top_left.y
        xmax = scene_configuration.backdrop.bottom_right.x
        ymax = scene_configuration.backdrop.bottom_right.y
        
        # resize
        gray_resized_cropped = gray_resized[ymin:ymax, xmin:xmax]

    else:
        gray_resized_cropped = gray_resized.copy()

    return gray_resized_cropped


### open video capture
# first example
video_file_path = "/home/saeed/Downloads/IMG_5115.MOV"
# ffmpeg -i IMG_5115.MOV -ss 00:00:14 -t 00:00:40 -c:v copy -c:a copy output1.mp4
video_file_path = "/home/saeed/Downloads/IMG_5115_trimmed.mp4"

# # second example 
# video_file_path = "/home/saeed/Downloads/2023_03_12_5ft_Silicone_OD_7_8_SN_View.MOV"
# # ffmpeg -i 2023_03_12_5ft_Silicone_OD_7_8_SN_View.MOV -ss 00:00:20 -t 00:00:50 -c:v copy -c:a copy 2023_03_12_5ft_Silicone_OD_7_8_SN_View_trimmed.mp4
# video_file_path = "/home/saeed/Downloads/2023_03_12_5ft_Silicone_OD_7_8_SN_View_trimmed.mp4"

# video_file_path = "/home/saeed/Downloads/2023_03_12_5ft_Silicone_OD_7_8_WE_View.MOV"
# # ffmpeg -i 2023_03_12_5ft_Silicone_OD_7_8_WE_View.MOV -ss 00:00:05 -t 00:00:35 -c:v copy -c:a copy 2023_03_12_5ft_Silicone_OD_7_8_WE_View_trimmed.mp4
# video_file_path = "/home/saeed/Downloads/2023_03_12_5ft_Silicone_OD_7_8_WE_View_trimmed.mp4"

# # 20230328
# video_file_path = "/home/saeed/Downloads/pipe_resonance-20230401T095353Z-001/pipe_resonance/20230328_3.8x3.4_tube_strobe4800fpm_camera240fps/South_North.MOV"
# # ffmpeg -i South_North.MOV -ss 00:00:06 -t 00:00:40 -c:v copy -c:a copy South_North_trimmed.mp4
# # video_file_path = "/home/saeed/Downloads/pipe_resonance-20230401T095353Z-001/pipe_resonance/20230328_3.8x3.4_tube_strobe4800fpm_camera240fps/South_North_trimmed.MOV"

# video_file_path = "/home/saeed/Downloads/pipe_resonance-20230401T095353Z-001/pipe_resonance/2023-04-03 at 02.36.31_upward.mp4"
# video_file_path = "/home/saeed/Downloads/2023.04.03. 5,8x7,8 tube, Upward Riser, 60 fps.MOV"


##### open video
cap = cv2.VideoCapture(video_file_path)

##### select points on the image to mark regions of interests
global scene_config
scene_config = SceneConfiguration()

def scene_configuration_callback(event, x, y, flags, params):
    _, _ = flags, params
    global scene_config
    if event == cv2.EVENT_LBUTTONDOWN:
        scene_config.set(Pixel(y=int(y), x=int(x)))
        print(scene_config)

# get the first frame and process it for
ret, frame = cap.read()
assert ret is True, "failed to read frame from video capture"
image_gray = _process_frame(frame, scene_config)
cv2.imshow("scene_config", image_gray)
cv2.setMouseCallback("scene_config", scene_configuration_callback)
cv2.waitKey(0)

# draw scene config
gray_resized_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
draw_scene_configuration(gray_resized_color, scene_config)
cv2.imshow("scene_config", gray_resized_color)
cv2.waitKey(0)
cv2.destroyWindow("scene_config")

assert scene_config.is_set(), "Cannot continue if all the points are not selected"

##### adaptive_binarization_threshold
adaptive_binarization_threshold = BINARIZATION_THRESHOLD
if False:
    ret, frame = cap.read()
    assert ret is True, "Did not get any frame"
    pipe_wrt_backdrop = scene_config.pipe_wrt_backdrop()
    pipe_col = (pipe_wrt_backdrop.top.col + pipe_wrt_backdrop.bottom.col) // 2
    pipe_region_initial = image_gray[
        pipe_wrt_backdrop.top.row : pipe_wrt_backdrop.bottom.row,
        pipe_col-PIPE_WIDTH_PIXEL : pipe_col+PIPE_WIDTH_PIXEL,
    ]
    adaptive_binarization_threshold = (numpy.median(pipe_region_initial) + numpy.median(image_gray)) // 2

##### main loop
deviations: List[Any] = []
play_single_frame = False
play = True
first_frame = True
skeleton_roi = None
while(cap.isOpened()):

    ###### playback control
    pressed_key = cv2.waitKey(30) & 0xFF
    play_single_frame = False
    if pressed_key == ord('q'):
        break
    if pressed_key == ord('p'):
        play = not play
    if pressed_key == 83: # right arrow
        play_single_frame = True
    if pressed_key == 81: # left arrow
        pass
    if not play and not play_single_frame:
        continue
    
    ##### read frame
    ret, frame = cap.read()
    if ret is False:
        break

    gray_resized_cropped = _process_frame(frame, scene_config)
    equalized = equalize(gray_resized_cropped, mode=EQUALIZATION_MODE, ksize=EQUALIZATION_KERNEL_SIZE)
    blurred = blur(equalized, mode=BLURRING_MODE, ksize=BLURRING_KERNEL_SIZE)
    binary = binarize(
        blurred, mode=BINARIZATION_MODE, ksize=BINARIZATION_KERNEL_SIZE, thresh=adaptive_binarization_threshold
    )
    (
        distance_img,
        laplace_img,
        skeleton_mask,
        skeleton_image,
        skeleton_image_eroded
    ) = skeletonize(binary, skeleton_roi, skeleton_radius=9, erosion_size=None)

    dilation_size = 35
    dilation_kernel = numpy.ones((dilation_size, dilation_size), numpy.uint8)
    skeleton_roi = cv2.dilate(skeleton_image_eroded, dilation_kernel, iterations=1)
    skeleton_points = _points_from_skeleton(skeleton_image_eroded)

    image_hieght, image_width = gray_resized_cropped.shape
    images = {
        "original with points": gray_resized_cropped,
        "equalized": equalized,
        "blurred": blurred,
        "binary": binary,
        "skeleton_roi": skeleton_roi,
        "skeleton_image_eroded": skeleton_image_eroded,
    }
    image_display = numpy.hstack([img for img in images.values()])
    image_display = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)
    for i, text in enumerate(images.keys()):
        location = Pixel(row=15, col=i * image_width + 1)
        image_display = write(image_display, text, location, BLUE)
    image_display = draw_points(image_display, skeleton_points, RED)
    cv2.imshow('display', image_display)

    deviations.append(skeleton_points[:, 1] - (scene_config.pipe.bottom.col - scene_config.backdrop.top_left.col))

# 
cap.release()
cv2.destroyAllWindows()

# draw deviations
for frame_idx, deviation in enumerate(deviations):
    plt.plot([frame_idx]* deviation.shape[0], deviation, "k,")
deviation_upper_bound = [deviation.max() for deviation in deviations]
deviation_lower_bound = [deviation.min() for deviation in deviations]
plt.plot(deviation_upper_bound, "r")
plt.plot(deviation_lower_bound, "r")
plt.xlabel("frame index")
plt.ylabel("horizontal deviation in pixel")
plt.show()
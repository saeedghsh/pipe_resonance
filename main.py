#!/usr/bin/env python
"""Main script to launch examples from."""

import numpy
import cv2
from typing import Tuple, Optional, List, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from user_defined_types import Pixel
from scene_configuration import SceneConfiguration
from image_morph import equalize, blur, binarize, skeletonize
from image_draw import RED, BLUE, draw_points, write, draw_scene_configuration, draw_lableled_image

RESIZE = 0.2  # resizing factor of the image

EQUALIZATION_MODE = ["histogram", "clahe", None, ][1]
EQUALIZATION_KERNEL_SIZE = 8

BLURRING_MODE = ["blur", "GaussianBlur", "medianBlur", "bilateralFilter", None, ][0]
BLURRING_KERNEL_SIZE = 15

BINARIZATION_MODE = ["thresh_binary", "thresh_otsu", "adaptive_thresh_gaussian", "adaptive_thresh_mean", None, ][1]
BINARIZATION_THRESHOLD = 100
BINARIZATION_KERNEL_SIZE = 9

PIPE_WIDTH_PIXEL = 10

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


def _nonzero_points_as_numpy_2darray_rc(skeleton: numpy.ndarray) -> numpy.ndarray:
    rows, cols = numpy.nonzero(skeleton)
    return numpy.array([rows, cols]).T


def _parametrize_pipe():
    # TODO:
    # treat it as a path planning problem
    # start from one end and traverse to the other end
    # distribute a predefined number of points uniformly
    return


def _process_frame(
    image: numpy.ndarray,
    scene_configuration: SceneConfiguration,
    convert_to_gray: bool,
    resize: Optional[float] = RESIZE,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if convert_to_gray else image.copy()

    if resize is not None:
        res = cv2.resize(res, (0,0), fx=resize, fy=resize)

    if scene_configuration.is_set():
        xmin = scene_configuration.backdrop.top_left.x
        ymin = scene_configuration.backdrop.top_left.y
        xmax = scene_configuration.backdrop.bottom_right.x
        ymax = scene_configuration.backdrop.bottom_right.y
        res = res[ymin:ymax, xmin:xmax]

    return res

def _find_pipe(skeleton_image: numpy.ndarray, scene_configuration: SceneConfiguration) -> numpy.ndarray:
    num_labels, labels = cv2.connectedComponents(skeleton_image)
    
    # pipe_component is not necessarily the biggest, but the component closet to the pipe top and bottom
    pipe_top_wrt_backdrop = scene_configuration.pipe_wrt_backdrop().top.as_numpy_2darray_rc()
    pipe_bottom_wrt_backdrop = scene_configuration.pipe_wrt_backdrop().bottom.as_numpy_2darray_rc()
    min_distance = 10e10
    selected_label = 0
    for label in range(1, num_labels):
        points = _nonzero_points_as_numpy_2darray_rc(labels == label)
        distance_matrix_to_pipe_top = cdist(pipe_top_wrt_backdrop, points)
        distance_matrix_to_pipe_bottom = cdist(pipe_bottom_wrt_backdrop, points)
        distance = distance_matrix_to_pipe_top.min() + distance_matrix_to_pipe_bottom.min()
        if distance < min_distance:
            min_distance = distance
            selected_label = label

    pipe_image = numpy.zeros(labels.shape).astype(int)
    if selected_label != 0:
        pipe_image[labels==selected_label] = 255
    return pipe_image, labels

##### open video
cap = cv2.VideoCapture(video_file_path)

##### select points on the image to mark regions of interests
global scene_configuration
scene_configuration = SceneConfiguration()

def scene_configuration_callback(event, x, y, flags, params):
    _, _ = flags, params
    global scene_configuration
    if event == cv2.EVENT_LBUTTONDOWN:
        scene_configuration.set(Pixel(y=int(y), x=int(x)))
        print(scene_configuration)

# get the first frame and process it for
ret, frame = cap.read()
assert ret is True, "failed to read frame from video capture"
image = _process_frame(frame, scene_configuration, convert_to_gray=False)
cv2.imshow("scene_config", image)
cv2.setMouseCallback("scene_config", scene_configuration_callback)
cv2.waitKey(0)

# draw scene config
draw_scene_configuration(image, scene_configuration)
cv2.imshow("scene_configuration", image)
cv2.waitKey(0)
cv2.destroyWindow("scene_configuration")

assert scene_configuration.is_set(), "Cannot continue if all the points are not selected"


##### main loop
deviations: List[Any] = []
play_single_frame = False
play = True
first_frame = True
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

    gray_resized_cropped = _process_frame(frame, scene_configuration, convert_to_gray=True)
    equalized = equalize(gray_resized_cropped, mode=EQUALIZATION_MODE, ksize=EQUALIZATION_KERNEL_SIZE)
    blurred = blur(equalized, mode=BLURRING_MODE, ksize=BLURRING_KERNEL_SIZE)
    binary = binarize(
        blurred, mode=BINARIZATION_MODE, ksize=BINARIZATION_KERNEL_SIZE, thresh=BINARIZATION_THRESHOLD
    )
    (
        _,  # distance_image,
        _,  # laplace_image,
        _,  # skeleton_mask,
        _,  # skeleton_image,
        skeleton_image_eroded
    ) = skeletonize(binary, skeleton_radius=9, erosion_size=None)
    pipe_image, labels = _find_pipe(skeleton_image_eroded, scene_configuration)
    pipe_points = _nonzero_points_as_numpy_2darray_rc(pipe_image)

    image_hieght, image_width = gray_resized_cropped.shape
    images = {
        "original with points": _process_frame(frame, scene_configuration, convert_to_gray=False),
        "equalized": cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR),
        "blurred": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "binary": cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "skeleton_image_eroded": cv2.cvtColor(skeleton_image_eroded, cv2.COLOR_GRAY2BGR),
        "connected components": draw_lableled_image(labels)
    }
    
    image_display = numpy.hstack([img for img in images.values()])
    for i, text in enumerate(images.keys()):
        location = Pixel(row=15, col=i * image_width + 1)
        image_display = write(image_display, text, location, BLUE)
    image_display = draw_points(image_display, pipe_points, RED)
    cv2.imshow('display', image_display)

    # TODO: this deviation estimate only works with vertical pipe assumption
    #       discarding the "vertical pipe assumption" begs a proper definition of "deviation".
    #       It can be defined in reference to the starting position, but along what exacly? horizontal or normal to pipe?
    deviation = pipe_points[:, 1] - (scene_configuration.pipe.bottom.col - scene_configuration.backdrop.top_left.col)
    if deviation.size > 0:
        deviations.append(deviation)


cap.release()
cv2.destroyAllWindows()

if len(deviations):
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
else:
    print("\nNo points were ever detected to compute deviation for\n")
from functools import partial
import numpy
import cv2


from user_defined_types import Color, Pixel
from scene_configuration import SceneConfiguration

# BGR
WHITE: Color = [255, 255, 255]
BLACK: Color = [0, 0, 0]
BLUE: Color = [255, 0, 0]
GREEN: Color = [0, 255, 0]
RED: Color = [0, 0, 255]

# # RGB
# YELLOW: Color = [255, 255, 0]	
# BLACK: Color = [0, 0, 0]
# WHITE: Color = [255, 255, 255]
# RED: Color = [255, 0, 0]
# LIME: Color = [0, 255, 0]
# BLUE: Color = [0, 0, 255]
# YELLOW: Color = [255, 255, 0]
# CYAN: Color = [0, 255, 255]
# MAGENTA: Color = [255, 255, 0]
# SILVER: Color = [192, 192, 192]
# GRAY: Color = [128, 128, 128]
# MAROON: Color = [128, 0, 0]
# OLIVE: Color = [128, 128, 0]
# GREEN: Color = [0, 128, 0]
# PURPLE: Color = [128, 0, 128]
# TEAL: Color = [0, 128, 128]
# NAVY: Color = [0, 0, 128]


def draw_points(
    image: numpy.ndarray, points: numpy.ndarray, color: Color
) -> numpy.ndarray:
    image[points[:, 0].astype(int), points[:, 1].astype(int)] = color
    return image


def write(image: numpy.ndarray, text: str, location: Pixel, color: Color) -> numpy.ndarray:
    """location is the bottom left corner of the text"""
    scale = 0.5
    thickness = 1
    line_type  = 2
    cv2.putText(
        image, text, (location.x, location.y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, line_type
    )
    return image


def draw_circle(
    image: numpy.ndarray, center: Pixel, radius: float, color: Color, thickness: float = 1,
) -> numpy.ndarray:
    cv2.circle(img=image, center=(center.x, center.y), radius=radius, color=color, thickness=thickness, lineType=cv2.FILLED)
    return image


def draw_cross(
        image: numpy.ndarray, center: Pixel, radius: float, color: Color, thickness: float = 1,
) -> numpy.ndarray:
    radius *= numpy.sqrt(2) / 2
    _draw_line = partial(cv2.line, img=image, color=color, thickness=thickness)
    
    top_left = Pixel(x=int(center.x - radius), y=int(center.y - radius))
    bottom_right = Pixel(x=int(center.x + radius), y=int(center.y + radius))
    _draw_line(pt1=(top_left.x, top_left.y), pt2=(bottom_right.x, bottom_right.y))

    top_right = Pixel(x=int(center.x + radius), y=int(center.y - radius))
    bottom_left = Pixel(x=int(center.x - radius), y=int(center.y + radius))
    _draw_line(pt1=(top_right.x, top_right.y), pt2=(bottom_left.x, bottom_left.y))
    return image


def draw_corssed_circle(
        image: numpy.ndarray, center: Pixel, radius: float, color: Color, thickness: float = 1,
) -> numpy.ndarray:
    draw_circle(image, center, radius, color, thickness)
    draw_cross(image, center, radius, color, thickness)


def draw_scene_configuration(
        image: numpy.ndarray,
        scene_configuration: SceneConfiguration,
) -> numpy.ndarray:
    rad = 10
    text_offset = Pixel(row=-rad, col=+rad)
    markers = {
        "backdrop.top_left": scene_configuration.backdrop.top_left,
        "backdrop.bottom_right": scene_configuration.backdrop.bottom_right,
        "checkboard.top_left": scene_configuration.checkerboard.top_left,
        "checkboard.bottom_right": scene_configuration.checkerboard.bottom_right,
        "strobe.center": scene_configuration.strobe.center,
        "pipe.top": scene_configuration.pipe.top,
        "pipe.bottom": scene_configuration.pipe.bottom,
    }
    for text, location in markers.items():
        if location is None:
            continue
        draw_corssed_circle(image, location, rad, RED)
        write(image, text, location + text_offset, RED)
    return image
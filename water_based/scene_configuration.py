""""""

from typing import Optional
from user_defined_types import Pixel

class AxisAlignedBox:
    def __init__(self, top_left: Optional[Pixel] = None, bottom_right: Optional[Pixel] = None):
        self._top_left = top_left
        self._bottom_right = bottom_right

    @property
    def top_left(self) -> Pixel:
        return self._top_left
    @property
    def bottom_right(self) -> Pixel:
        return self._bottom_right
    @top_left.setter
    def top_left(self, pixel: Pixel):
        self._top_left = pixel
    @bottom_right.setter
    def bottom_right(self, pixel: Pixel):
        self._bottom_right = pixel


class Strobe:
    def __init__(self, center: Optional[Pixel] = None, radius: Optional[int] = 5):
        self._center = center
        self._radius = radius
    @property
    def center(self) -> Pixel:
        return self._center
    @property
    def radius(self) -> int:
        return self._radius
    @center.setter
    def center(self, pixel: Pixel):
        self._center = pixel
    @radius.setter
    def radius(self, radius: int):
        self._radius = radius


class Pipe:
    def __init__(self, top: Pixel = None, bottom: Pixel = None):
        self._top = top
        self._bottom = bottom
    @property
    def top(self) -> Pixel:
        return self._top
    @property
    def bottom(self) -> Pixel:
        return self._bottom
    @top.setter
    def top(self, pixel: Pixel):
        self._top = pixel
    @bottom.setter
    def bottom(self, pixel: Pixel):
        self._bottom = pixel


# TODO: refactor this shit! the logic of this (is_set and set) and the cv2 operation are mixed in the callback function
# TODO: either remover the getters for strobe and pipe (that are not wrt backdrop) or issue a warning! somehow!
class SceneConfiguration:
    def __init__(self, backdrop = AxisAlignedBox(), strobe = Strobe(), pipe = Pipe(), checkerboard = AxisAlignedBox()):
        self._backdrop = backdrop
        self._strobe = strobe
        self._pipe = pipe
        self._checkerboard = checkerboard

    @property
    def backdrop(self) -> AxisAlignedBox:
        return self._backdrop
    @property
    def strobe(self) -> Strobe:
        return self._strobe
    @property
    def pipe(self) -> Pipe:
        return self._pipe
    @property
    def checkerboard(self) -> AxisAlignedBox:
        return self._checkerboard
    @backdrop.setter
    def top(self, backdrop: AxisAlignedBox):
        self._backdrop = backdrop
    @strobe.setter
    def strobe(self, strobe: Strobe):
        self._strobe = strobe
    @pipe.setter
    def pipe(self, pipe: Pipe):
        self._pipe = pipe
    @checkerboard.setter
    def checkerboard(self, checkerboard: AxisAlignedBox):
        self._checkerboard = checkerboard

    def __repr__(self) -> str:
        return "\n".join(
            [
                "\n",
                "ALL SET - MOVE ON\n" if self.is_set() else "",
                f"backdrop top_left         {self.backdrop.top_left}",
                f"backdrop bottom_right     {self.backdrop.bottom_right}",
                f"checkerboard top_left     {self.checkerboard.top_left}",
                f"checkerboard bottom_right {self.checkerboard.bottom_right}",
                f"strobe center             {self.strobe.center}",
                f"pipe top                  {self.pipe.top}",
                f"pipe bottom               {self.pipe.bottom}",
            ]
        )

    def is_set(self) -> bool:
        if self.backdrop.top_left is None:
            return False
        if self.backdrop.bottom_right is None:
            return False
        if self.strobe.center is None:
            return False
        if self.pipe.top is None:
            return False
        if self.pipe.bottom is None:
            return False
        return True

    def set(self, pixel: Pixel):
        if self.backdrop.top_left is None:
            self.backdrop.top_left = pixel
            return
        if self.backdrop.bottom_right is None:
            self.backdrop.bottom_right = pixel
            return
        if self.checkerboard.top_left is None:
            self.checkerboard.top_left = pixel
            return
        if self.checkerboard.bottom_right is None:
            self.checkerboard.bottom_right = pixel
            return
        if self.strobe.center is None:
            self.strobe.center = pixel
            return
        if self.pipe.top is None:
            self.pipe.top = pixel
            return
        if self.pipe.bottom is None:
            self.pipe.bottom = pixel

    def pixel_wrt_backdrop(self, pixel: Pixel) -> Pixel:
        return Pixel(
            row=pixel.row - self.backdrop.top_left.row,
            col=pixel.col - self.backdrop.top_left.col
        )

    def pipe_wrt_backdrop(self) -> Pipe:
        return Pipe(
            self.pixel_wrt_backdrop(self.pipe.top),
            self.pixel_wrt_backdrop(self.pipe.bottom)
        )

    def strobe_wrt_backdrop(self) -> Strobe:
        return Strobe(self.pixel_wrt_backdrop(self.strobe.center))


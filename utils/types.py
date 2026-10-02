from collections.abc import Sequence
from typing import NamedTuple, TypedDict

from cv2.typing import MatLike


class HistCfg(TypedDict, total=False):
    show_hist: bool
    log_scale: bool


class BoundingBox(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int



class AutoCropResult(TypedDict):
  images: dict[str, MatLike]
  contours: Sequence[MatLike]
  valid_contours: Sequence[MatLike]
  bbox: BoundingBox | None
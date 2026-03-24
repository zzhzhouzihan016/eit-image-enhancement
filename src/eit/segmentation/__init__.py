"""CT / MedSAM 分割模块。"""

from .bbox import find_lung_bbox, find_torso_bbox
from .medsam import MedSAMSegmenter, segment_torso_and_lungs

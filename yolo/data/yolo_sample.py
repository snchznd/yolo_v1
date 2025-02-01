from dataclasses import dataclass
from typing import List

import torchvision
import torchvision.tv_tensors


@dataclass
class YoloSample:
    image: torchvision.tv_tensors.Image
    class_labels: List[int]
    bounding_boxes: torchvision.tv_tensors.BoundingBoxes # format: [x,y,w,h]

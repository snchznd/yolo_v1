from typing import List

import torch
import torchvision.transforms.v2

from .yolo_sample import YoloSample


def transform_yolo_sample(
    yolo_sample: YoloSample, transforms_list: List[torch.nn.Module]
) -> YoloSample:  # inplace
    transforms_composition = torchvision.transforms.v2.Compose(transforms_list)
    yolo_sample.image, yolo_sample.bounding_boxes = transforms_composition(
        yolo_sample.image, yolo_sample.bounding_boxes
    )
    return yolo_sample

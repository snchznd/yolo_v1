import matplotlib.patches as patches
import torch
from matplotlib import pyplot as plt

from .yolo_sample import YoloSample


def get_yolo_bounding_box_corner_coordinates(
    bounding_box: torch.tensor,
) -> torch.tensor:
    bb_center_x, bb_center_y, bb_width, bb_height = bounding_box
    bb_corner_x = bb_center_x - bb_width / 2
    bb_corner_y = bb_center_y - bb_height / 2
    return bb_corner_x.item(), bb_corner_y.item(), bb_width.item(), bb_height.item()


def plot_yolo_sample(yolo_sample: YoloSample, hide_axis: bool = False) -> None:
    fig, ax = plt.subplots()
    ax.imshow(yolo_sample.image.permute(1, 2, 0))

    for bb_tensor in yolo_sample.bounding_boxes:
        x, y, w, h = get_yolo_bounding_box_corner_coordinates(
            bb_tensor.detach().clone()
        )
        rect = patches.Rectangle(
            xy=(x, y), width=w, height=h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    if hide_axis:
        ax.axis("off")
    plt.show()

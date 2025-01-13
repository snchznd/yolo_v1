import matplotlib.axes
import matplotlib.patches as patches
import matplotlib
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


def plot_yolo_sample(
    yolo_sample: YoloSample, hide_axis: bool = False, show_grid=False, show_center : bool = True
) -> None:
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
        if show_center:
            ax.scatter(bb_tensor[0], bb_tensor[1], color='magenta', s=50)
    if show_grid:
        draw_grid(
            ax=ax, yolo_sample=yolo_sample, nbr_horizontal_cells=7, nbr_vertical_cells=7
        )
    if hide_axis:
        ax.axis("off")
    plt.show()

def draw_grid(
    ax: matplotlib.axes,
    yolo_sample: YoloSample,
    nbr_horizontal_cells: int,
    nbr_vertical_cells: int,
) -> None:
    image_height = yolo_sample.image.shape[1]
    image_width = yolo_sample.image.shape[2]
    grid_cell_height = int(image_height / nbr_vertical_cells)
    grid_cell_width = int(image_width / nbr_horizontal_cells)

    # drawing horizontal lines
    vertical_position = 0
    for horizontal_line in range(1, nbr_vertical_cells):
        vertical_position += grid_cell_height
        # draw horizontal line at that height
        ax.axhline(y=vertical_position, color="lime", linewidth=2, linestyle='-')

    horizontal_position = 0
    # drawing vertical lines
    for vertical_line in range(1, nbr_horizontal_cells):
        horizontal_position += grid_cell_width
        # draw vertical line at that horizontal position
        ax.axvline(x=horizontal_position, color="lime", linewidth=2, linestyle='-')
    plt.show()
    
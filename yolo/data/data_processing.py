from typing import List, Tuple

import torch
import torchvision.transforms.v2
import torchvision.tv_tensors

from .yolo_sample import YoloSample


def transform_yolo_sample(
    yolo_sample: YoloSample, transforms_list: List[torch.nn.Module]
) -> YoloSample:  # inplace
    transforms_composition = transforms_list
    if not isinstance(transforms_list, torchvision.transforms.v2._container.RandomApply):
        transforms_composition = torchvision.transforms.v2.Compose(transforms_list)
    yolo_sample.image, yolo_sample.bounding_boxes = transforms_composition(
        yolo_sample.image, yolo_sample.bounding_boxes
    )
    return yolo_sample


def get_grid_cell(
    image: torchvision.tv_tensors.Image,
    bounding_box_center_y: int,
    bounding_box_center_x: int,
    S_height: int,
    S_width: int,
) -> Tuple[int, int]:
    image_height = image.shape[1]
    image_width = image.shape[2]
    grid_cell_height = int(image_height / S_height)
    grid_cell_width = int(image_width / S_width)
    bounding_box_grid_x = bounding_box_center_x // grid_cell_width
    bounding_box_grid_y = bounding_box_center_y // grid_cell_height
    
    # We need the operations bellow in case we want to use rotations for data
    # augmentation as they may put the center of the bb of an object at the 
    # edge of the grid which would put it in grid cell S. This is a problem
    # because we only have grid cells 0, 1, ..., S-1 so it would cause an
    # IndexError
    bounding_box_grid_x = min(bounding_box_grid_x, S_width-1)
    bounding_box_grid_y = min(bounding_box_grid_y, S_height-1)
    
    return int(bounding_box_grid_y), int(bounding_box_grid_x)

def get_normalized_center_coordinates(
    image: torchvision.tv_tensors.Image,
    bounding_box_center_y: int,
    bounding_box_center_x: int,
    S_height: int,
    S_width: int,
) -> Tuple[int,int]:
    image_height = image.shape[1]
    image_width = image.shape[2]
    grid_cell_height = int(image_height / S_height)
    grid_cell_width = int(image_width / S_width)
    normalized_y = (bounding_box_center_y % grid_cell_height) / grid_cell_height
    normalized_x = (bounding_box_center_x % grid_cell_width) / grid_cell_width
    return normalized_y, normalized_x

def create_target_tensor(
    yolo_sample: YoloSample, S_height: int = 7, S_width: int = 7, C: int = 20
) -> torch.tensor:
    target_tensor = torch.zeros(size=(S_height, S_width, C + 5))
    for bounding_box, class_label in zip(yolo_sample.bounding_boxes, yolo_sample.class_labels):
        # format: [x,y,w,h,c,p_1,...,p_C]
        
        # 1. compute which grid cell is responsible for the BB
        grid_y, grid_x = get_grid_cell(
            image=yolo_sample.image,
            bounding_box_center_y=bounding_box[1].item(),
            bounding_box_center_x=bounding_box[0].item(),
            S_height=S_height,
            S_width=S_width,
        )
        
        # 2. compute the normalized cx, xy (w.r.t. the cell dimension)
        y, x = bounding_box[1].item(), bounding_box[0].item()
        normalized_y, normalized_x = get_normalized_center_coordinates(
            image=yolo_sample.image,
            bounding_box_center_y=y,
            bounding_box_center_x=x,
            S_height=S_height,
            S_width=S_width
        )
        #print(grid_y, grid_x)
        target_tensor[grid_y,grid_x,0] = normalized_x
        target_tensor[grid_y,grid_x,1] = normalized_y
        
        # 3. compute the normalized w, h (w.r.t. the image dimension)
        normalized_w = bounding_box[2] / yolo_sample.image.shape[2]
        normalized_h = bounding_box[3] / yolo_sample.image.shape[1]
        target_tensor[grid_y,grid_x,2] = normalized_w
        target_tensor[grid_y,grid_x,3] = normalized_h
        
        # 4. create one-hot vector of dim C = 20 with the right label
        #print(f'{class_label=}')
        target_tensor[grid_y,grid_x, class_label+5] = 1
        
        # 5. confidence probability score
        target_tensor[grid_y,grid_x,4] = 1
        
        #print(f"{grid_y=} {grid_x=}")
        #print(f"y={normalized_y*100:.2f}% ; x={normalized_x*100:.2f}%\n")
    return target_tensor 

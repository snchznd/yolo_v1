import os
import random
from typing import List, Tuple, Optional

import pandas as pd
import torch
import torchvision

from yolo.data.yolo_sample import YoloSample


def load_image_tensor(
    image_file_name: str, images_dir: str,
) -> torch.tensor:
    image_path = os.path.join(images_dir, image_file_name)
    return torchvision.tv_tensors.Image(
        torchvision.io.read_image(image_path), dtype=torch.float32
    )


def load_labels(
    label_file_name: str, labels_dir: str
) -> Tuple[List[int], List[torch.tensor]]:
    label_path = os.path.join(labels_dir, label_file_name)
    with open(file=label_path, mode="r", encoding="utf-8") as f:
        file_content = f.read()
    labels_list = file_content.split("\n")[:-1]
    class_labels_arr = []
    bounding_boxes_arr = []
    for label_string in labels_list:
        class_label, *bounding_box = label_string.split(" ")
        bounding_box = [float(x) for x in bounding_box]
        class_labels_arr.append(int(class_label))
        bounding_boxes_arr.append(bounding_box)
    return class_labels_arr, bounding_boxes_arr


def get_sample(sample_idx: int, df_mapping: pd.DataFrame) -> Tuple[str, str]:
    # get x, y pairs from df containing mapping. E.g. a row is: (image_file_path, labels_file_path).
    row = df_mapping.iloc[sample_idx]
    return row.iloc[0], row.iloc[1]


def load_yolo_sample(sample_idx: int, df_mapping: pd.DataFrame, images_dir : str, labels_dir : str) -> YoloSample:
    image_file, label_file = get_sample(sample_idx, df_mapping)
    image_tensor = load_image_tensor(image_file_name=image_file,images_dir=images_dir)  # shape (C, H, W)
    class_labels_arr, bounding_boxes_arr = load_labels(label_file, labels_dir=labels_dir)

    # Convert bounding boxes from normalized to absolute pixel coords
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]

    bounding_boxes_abs = []
    for bb in bounding_boxes_arr:  # each bb is [cx, cy, w, h]
        cx = bb[0] * width
        cy = bb[1] * height
        w = bb[2] * width
        h = bb[3] * height
        bounding_boxes_abs.append([cx, cy, w, h])

    # Now create a BoundingBoxes object with absolute coords
    bounding_boxes = torchvision.tv_tensors.BoundingBoxes(
        data=bounding_boxes_abs,
        format=torchvision.tv_tensors.BoundingBoxFormat.CXCYWH,
        canvas_size=(height, width),  # H, W
    )
    return YoloSample(
        image=image_tensor, class_labels=class_labels_arr, bounding_boxes=bounding_boxes
    )


def load_random_yolo_sample(
    df_mapping: pd.DataFrame,
    images_dir : str,
    labels_dir : str
) -> Tuple[torch.tensor, List[torch.tensor]]:
    sample_idx = random.randint(0, len(df_mapping) - 1)
    return load_yolo_sample(sample_idx, df_mapping, images_dir=images_dir, labels_dir=labels_dir)


def generate_train_val_partitions(
    original_train_set_mapping_path : str,
    train_set_mapping_path : str,
    validation_set_mapping_path : str,
    validation_fraction: float = 0.1,
    random_state: Optional[int] = None
) -> None:
    # load original train dataset
    df_train_original = pd.read_csv(original_train_set_mapping_path, header=None)

    # split it randomly into train and val subsets
    validation_split = df_train_original.sample(
        frac=validation_fraction, random_state=random_state
    )
    train_split = df_train_original.drop(validation_split.index)

    # save train and val
    train_split.to_csv(train_set_mapping_path, index=False, header=False)
    validation_split.to_csv(validation_set_mapping_path, index=False, header=False)

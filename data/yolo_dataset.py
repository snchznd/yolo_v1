from typing import List, Tuple

import pandas as pd
import torch
import torch.utils.data.dataset
import torchvision
import torchvision.tv_tensors

from .data_loading import (
    TEST_SET_MAPPING_PATH,
    TRAIN_SET_MAPPING_PATH,
    VALIDATION_SET_MAPPING_PATH,
    load_yolo_sample,
)
from .data_processing import create_target_tensor, transform_yolo_sample
from .dataset_split import DatasetSplit
from .yolo_sample import YoloSample

TRAIN_TRANSFORMS_LIST = [
    torchvision.transforms.v2.Resize(size=(448,) * 2),
    torchvision.transforms.v2.RandomPhotometricDistort(p=0.25),
    torchvision.transforms.v2.RandomHorizontalFlip(p=0.25),
    torchvision.transforms.v2.RandomRotation(degrees=(-90, 90)),
]

EVAL_TRANSFORMS_LIST = [
    torchvision.transforms.v2.Resize(size=(448,) * 2)
]

class YoloDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_split: DatasetSplit,
        nbr_vertical_cells: int = 7,
        nbr_horizontal_cells: int = 7,
        nbr_classes: int = 20,
    ) -> None:

        if not isinstance(dataset_split, DatasetSplit):
            raise TypeError(
                f"Argument dataset_split must be of type DatasetSplit but is of type: {type(dataset_split)}."
            )

        match dataset_split:
            case DatasetSplit.TRAIN:
                self.df_mapping = pd.read_csv(TRAIN_SET_MAPPING_PATH)
                self.transforms_list = TRAIN_TRANSFORMS_LIST
            case DatasetSplit.VALIDATION:
                self.df_mapping = pd.read_csv(VALIDATION_SET_MAPPING_PATH)
                self.transforms_list = EVAL_TRANSFORMS_LIST
            case DatasetSplit.TEST:
                self.df_mapping = pd.read_csv(TEST_SET_MAPPING_PATH)
                self.transforms_list = EVAL_TRANSFORMS_LIST

        self.dataset_split = dataset_split
        self.nbr_horizontal_cells = nbr_horizontal_cells
        self.nbr_vertical_cells = nbr_vertical_cells
        self.nbr_classes = nbr_classes

    def __getitem__(
        self, idx: int
    ) -> Tuple[torchvision.tv_tensors.Image, torch.tensor]:
        yolo_sample = load_yolo_sample(sample_idx=idx, df_mapping=self.df_mapping)
        transform_yolo_sample(
            yolo_sample=yolo_sample, transforms_list=self.transforms_list
        )
        image = yolo_sample.image
        target_tensor = create_target_tensor(
            yolo_sample=yolo_sample,
            S_height=self.nbr_vertical_cells,
            S_width=self.nbr_horizontal_cells,
            C=self.nbr_classes,
        )
        return image, target_tensor

    def __len__(self):
        return len(self.df_mapping)

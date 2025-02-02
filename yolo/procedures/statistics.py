import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.data.dataset_split import DatasetSplit
from yolo.data.yolo_dataset import YoloDataset
from yolo.utils.config import load_config


def compute_channels_statistics(
    dataset: YoloDataset,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    total_sum = torch.zeros(3, dtype=torch.float64).to(device)
    total_sq = torch.zeros(3, dtype=torch.float64).to(device)
    total_pixels = 0

    # Compute total sum and sum of squares
    for imgs, _ in tqdm(data_loader, desc="Computing statistics", colour="magenta"):
        imgs = imgs.to(device)
        total_sum += imgs.sum(dim=(0, 2, 3))
        total_sq += (imgs**2).sum(dim=(0, 2, 3))
        total_pixels += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]

    mean = total_sum / total_pixels
    std = torch.sqrt(total_sq / total_pixels - mean**2)

    return mean, std


def save_channels_statistics(config_path: str) -> None:
    config_path = os.path.expanduser(config_path)
    config = load_config(config_path)

    datasets_kwargs = {
        "train_set_mapping_path": config["paths"]["train_set_mapping_path"],
        "validation_set_mapping_path": config["paths"]["validation_set_mapping_path"],
        "test_set_mapping_path": config["paths"]["test_set_mapping_path"],
        "images_dir": config["directories"]["images_dir"],
        "labels_dir": config["directories"]["labels_dir"],
    }

    dataset = YoloDataset(DatasetSplit.TRAIN, **datasets_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean, std = compute_channels_statistics(dataset=dataset, device=device)

    save_path = os.path.expanduser(config["directories"]["stats_dir"])
    torch.save(obj=mean, f=os.path.join(save_path, "mean.pth"))
    torch.save(obj=std, f=os.path.join(save_path, "std.pth"))


if __name__ == "__main__":
    save_channels_statistics(os.path.expanduser("~/projects/yolo/config.yaml"))

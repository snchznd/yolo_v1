import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yolo.data.data_loading import generate_train_val_partitions
from yolo.data.dataset_split import DatasetSplit
from yolo.data.utils import load_model
from yolo.data.yolo_dataset import YoloDataset
from yolo.loss.yolo_loss import YoloLoss
from yolo.model.yolo import YoloModel
from yolo.procedures.train import train
from yolo.utils.config import load_config


def add_train_cmd(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser("train", help="Launch a training session.")
    train_parser.add_argument(
        "-lbm",
        "--load_best_model",
        required=False,
        action="store_true",
        default=False,
        help="Whether to load the previous best model or start from zero.",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        action="store",
        type=str,
        help="The path to the config file.",
    )


def launch_train_procedure(args: argparse.Namespace) -> None:
    config = load_config(config_path=args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_train_val_partitions(
        validation_fraction=config["data"]["validation_fraction"]
    )
    train_dataset = YoloDataset(DatasetSplit.TRAIN)
    val_dataset = YoloDataset(DatasetSplit.VALIDATION)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["train_batch_size"],
        num_workers=config["data"]["loaders_num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_batch_size"],
        num_workers=config["data"]["loaders_num_workers"],
    )

    print(args.load_best_model)
    model = None
    if args.load_best_model:
        model = load_model(
            weights_path=os.path.join(
                config["paths"]["model_save_path"], "best_model.pth"
            ),
            device=device,
        )
    else:
        model = YoloModel()

    yolo_loss = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    writer = SummaryWriter(config["paths"]["tensorboard_log_path"])
    return None
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=yolo_loss,
        optimizer=optimizer,
        nbr_epochs=config["train"]["nbr_epochs"],
        writer=writer,
        device=device,
        perform_validation=config["train"]["perform_validation"],
        train_batch_logging_path=config["paths"]["train_batch_logging_path"],
        train_epoch_logging_path=config["paths"]["train_epoch_logging_path"],
        eval_batch_logging_path=config["paths"]["eval_batch_logging_path"],
        eval_epoch_logging_path=config["paths"]["eval_epoch_logging_path"],
        events_logging_path=config["paths"]["events_logging_path"],
        model_save_path=config["paths"]["model_save_path"],
    )

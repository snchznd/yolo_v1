import argparse
import datetime
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
        original_train_set_mapping_path=config["paths"]["original_train_set_mapping_path"],
        train_set_mapping_path=config["paths"]["train_set_mapping_path"],
        validation_set_mapping_path=config["paths"]["validation_set_mapping_path"],
        validation_fraction=config["data"]["validation_fraction"]
    )
    
    datasets_kwargs = {
        "train_set_mapping_path": config["paths"]["train_set_mapping_path"],
        "validation_set_mapping_path": config["paths"]["validation_set_mapping_path"],
        "test_set_mapping_path": config["paths"]["test_set_mapping_path"],
        "images_dir": config["directories"]["images_dir"],
        "labels_dir": config["directories"]["labels_dir"]
    }
    
    train_dataset = YoloDataset(dataset_split=DatasetSplit.TRAIN,
                                **datasets_kwargs)
    val_dataset = YoloDataset(dataset_split=DatasetSplit.VALIDATION, **datasets_kwargs)

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
    model = None
    if args.load_best_model:
        model = load_model(
            weights_path=config["paths"]["initial_model_path"],
            device=device,
        )
    else:
        model = YoloModel().to(device)

    yolo_loss = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    
    experiment_name = datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    
    # create directory for tensorboard logging
    tensorboar_logging_dir = os.path.join(os.path.expanduser(config["paths"]["tensorboard_log_path"]), experiment_name)
    try:
        os.makedirs(tensorboar_logging_dir)
    except OSError as e:
        print(f"Error creating directory: {e}")
        
    writer = SummaryWriter(tensorboar_logging_dir)
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=yolo_loss,
        optimizer=optimizer,
        nbr_epochs=config["train"]["nbr_epochs"],
        experiment_name=experiment_name,
        model_save_dir=config["directories"]["model_save_dir"],
        statistics_path=config["directories"]["stats_dir"],
        writer=writer,
        device=device,
        perform_validation=config["train"]["perform_validation"],
        train_batch_logging_path=config["directories"]["train_batch_logging_dir"],
        train_epoch_logging_path=config["directories"]["train_epoch_logging_dir"],
        eval_batch_logging_path=config["directories"]["eval_batch_logging_dir"],
        eval_epoch_logging_path=config["directories"]["eval_epoch_logging_dir"],
        events_logging_path=config["directories"]["events_logging_dir"],
        model_save_path=config["directories"]["model_save_dir"],
    )

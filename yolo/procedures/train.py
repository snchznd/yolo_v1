import datetime
import math
import os
from typing import Optional

import torch
from tqdm import tqdm

from yolo.procedures.eval import evaluate_model
from yolo.utils.loggers import LOGGING_PATH, get_file_logger


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    nbr_epochs: int,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
    device: str = "cuda",
    perform_validation: bool = True,
    train_batch_logging_path: Optional[str] = None,
    train_epoch_logging_path: Optional[str] = None,
    eval_batch_logging_path: Optional[str] = None,
    eval_epoch_logging_path: Optional[str] = None,
    events_logging_path: Optional[str] = None,
    model_save_path: Optional[str] = None,
) -> None:
    if train_batch_logging_path:
        train_batch_logger = get_file_logger(
            logger_name="train_batch",
            log_file=train_batch_logging_path,
        )

    if train_epoch_logging_path:
        train_epoch_logger = get_file_logger(
            logger_name="train_epoch",
            log_file=train_epoch_logging_path,
        )
    if eval_batch_logging_path:
        eval_batch_logger = get_file_logger(
            logger_name="evaluate_batch",
            log_file=eval_batch_logging_path,
        )
    if eval_epoch_logging_path:
        eval_epoch_logger = get_file_logger(
            logger_name="evaluate_epoch",
            log_file=eval_epoch_logging_path,
        )
    if events_logging_path:
        events_logger = get_file_logger(
            logger_name="event_logger",
            log_file=events_logging_path,
        )
    if perform_validation and not model_save_path:
        raise ValueError
    batch_counter = 0
    eval_loss = math.inf
    for epoch in range(nbr_epochs):
        epoch_losses = []
        model.train()
        for idx, (images, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader), colour="green"
        ):

            if epoch == 0 and idx == 0:
                batch_size = images.shape[0]

            # move tensors to the right device
            images, targets = images.to(device), targets.to(device)

            # zero the gradient
            optimizer.zero_grad()

            # forward pass
            predictions = model(images)
            loss = loss_func(predictions, targets)

            # backward pass
            loss.backward()

            # optimization step
            optimizer.step()

            # logging
            # writer.add_scalar("Loss/train", loss.detach().item(), epoch)
            batch_loss = loss.detach().item() / batch_size
            train_batch_logger.info(
                f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {batch_loss:>6.4f}"
            )
            epoch_losses.append(batch_loss)
            if writer:
                writer.add_scalar("train batch loss", batch_loss, batch_counter)
                batch_counter += 1

        # log epoch loss
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_epoch_logger.info(f"epoch: {epoch:>2} | loss: {epoch_loss:>6.4f}")

        if writer:
            writer.add_scalar("train epoch loss", epoch_loss, epoch)

        if perform_validation:
            # evaluate model on validation set
            epoch_eval_loss = evaluate_model(
                model,
                val_loader,
                loss_func,
                device,
                epoch,
                eval_batch_logger,
                eval_epoch_logger,
                writer,
            )
            if epoch_eval_loss < eval_loss:
                events_logger.info(
                    f"epoch: {epoch} | saving new best model with evaluation loss: {epoch_eval_loss:>6.4f}"
                )
                eval_loss = epoch_eval_loss
                # model_file_path = "best_model_" + datetime.datetime.now().strftime(
                #     "%d-%m-%y_%H:%M:%S"
                # ) + '.pth'
                model_file_path = "best_model_try"
                torch.save(
                    model.state_dict(), os.path.join(model_save_path, model_file_path)
                )

    if writer:
        writer.flush()

    # save last model
    last_model_file_path = "last_model.pth"
    last_model_save_path = os.path.join(model_save_path, last_model_file_path)
    torch.save(model.state_dict(), last_model_save_path)

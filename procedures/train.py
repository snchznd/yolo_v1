from tqdm import tqdm
import logging
import datetime
import time
import os
from .eval import evaluate_model
from utils.loggers import get_file_logger

LOGGING_PATH = "/home/masn/projects/yolo/logs"
EPOCH_LOGGING_PATH = os.path.join(LOGGING_PATH, 'train_epoch')
BATCH_LOGGING_PATH = os.path.join(LOGGING_PATH, 'train_batch')


def train(
    model,
    train_loader,
    val_loader,
    loss_func,
    optimizer,
    nbr_epochs,
    #writer,
    device: str = "cuda",
) -> None:
    batch_logger = get_file_logger(logger_name='train_batch',
                             log_file=BATCH_LOGGING_PATH,)
    epoch_logger = get_file_logger(logger_name='train_epoch',
                             log_file=EPOCH_LOGGING_PATH,)
    for epoch in range(nbr_epochs):
        epoch_losses = []
        model.train()
        for idx, (images, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
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
            #writer.add_scalar("Loss/train", loss.detach().item(), epoch)
            batch_loss = loss.detach().item() / batch_size
            batch_logger.info(
                f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {batch_loss:>6.3f}"
            )
            epoch_losses.append(batch_loss)

        # log epoch loss
        epoch_logger.info(
            f"epoch: {epoch:>2} | loss: {sum(epoch_losses) / len(epoch_losses) :>6.3f}"
        )
        
        # evaluate model on validation set
        #evaluate_model(model, val_loader, loss_func, device, epoch)


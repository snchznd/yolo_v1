import os

from tqdm import tqdm

from utils.loggers import LOGGING_PATH, get_file_logger

from .eval import evaluate_model

TRAIN_EPOCH_DIR = "train_epoch"
TRAIN_BATCH_DIR = "train_batch"
TRAIN_EPOCH_FILE_NAME = "train_epoch"
TRAIN_BATCH_FILE_NAME = "train_batch"
TRAIN_EPOCH_LOGGING_PATH = os.path.join(
    LOGGING_PATH, TRAIN_EPOCH_DIR, TRAIN_EPOCH_FILE_NAME
)
TRAIN_BATCH_LOGGING_PATH = os.path.join(
    LOGGING_PATH, TRAIN_BATCH_DIR, TRAIN_BATCH_FILE_NAME
)
EVAL_EPOCH_DIR = "eval_epoch"
EVAL_BATCH_DIR = "eval_batch"
EVAL_EPOCH_FILE_NAME = "eval_epoch"
EVAL_BATCH_FILE_NAME = "eval_batch"
EVAL_EPOCH_LOGGING_PATH = os.path.join(
    LOGGING_PATH, EVAL_EPOCH_DIR, EVAL_EPOCH_FILE_NAME
)
EVAL_BATCH_LOGGING_PATH = os.path.join(
    LOGGING_PATH, EVAL_BATCH_DIR, EVAL_BATCH_FILE_NAME
)


def train(
    model,
    train_loader,
    val_loader,
    loss_func,
    optimizer,
    nbr_epochs,
    writer=None,
    device: str = "cuda",
) -> None:
    train_batch_logger = get_file_logger(
        logger_name="train_batch",
        log_file=TRAIN_BATCH_LOGGING_PATH,
    )
    train_epoch_logger = get_file_logger(
        logger_name="train_epoch",
        log_file=TRAIN_EPOCH_LOGGING_PATH,
    )
    eval_batch_logger = get_file_logger(
        logger_name="evaluate_batch",
        log_file=EVAL_BATCH_LOGGING_PATH,
    )
    eval_epoch_logger = get_file_logger(
        logger_name="evaluate_epoch",
        log_file=EVAL_EPOCH_LOGGING_PATH,
    )
    batch_counter = 0
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
                f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {batch_loss:>6.3f}"
            )
            epoch_losses.append(batch_loss)
            if writer:
                writer.add_scalar("train batch loss", batch_loss, batch_counter)
                batch_counter += 1

        # log epoch loss
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_epoch_logger.info(f"epoch: {epoch:>2} | loss: {epoch_loss:>6.3f}")

        if writer:
            writer.add_scalar("train epoch loss", epoch_loss, epoch)

        # evaluate model on validation set
        evaluate_model(
            model,
            val_loader,
            loss_func,
            device,
            epoch,
            eval_batch_logger,
            eval_epoch_logger,
            writer,
        )

    if writer:
        writer.flush()

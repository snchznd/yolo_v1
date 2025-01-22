from tqdm import tqdm
import logging
import datetime
import time
import os
from .eval import evaluate_model

LOGGING_PATH = "/home/masn/projects/yolo/logs"


def train(
    model,
    train_loader,
    val_loader,
    loss_func,
    optimizer,
    nbr_epochs,
    writer,
    device: str = "cuda",
) -> None:
    for epoch in range(nbr_epochs):
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
            writer.add_scalar("Loss/train", loss.detach().item(), epoch)
            now = time.perf_counter()
            train_logger.info(
                f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {loss.detach().item() / batch_size :>6.3f}"
            )

        # evaluate model on validation set
        evaluate_model(model, val_loader, loss_func, device, epoch)


def create_logger():
    file_name = (
        "training_" + datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S") + ".log"
    )
    file_path = os.path.join(LOGGING_PATH, file_name)
    global train_logger
    train_logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=file_path,
        encoding="utf-8",
        level=logging.DEBUG,
        format="[%(asctime)s] - %(levelname)s - | %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )


create_logger()

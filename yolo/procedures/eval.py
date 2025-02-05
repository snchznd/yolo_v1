import logging

import torch
import torch.utils.tensorboard
from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    device: str,
    epoch: int,
    batch_logger: logging.Logger,
    epoch_logger: logging.Logger,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    mean : torch.tensor,
    std : torch.tensor,
) -> float:

    batch_losses = []
    model.eval()

    for idx, (images, targets) in tqdm(
        enumerate(data_loader), total=len(data_loader), colour="red"
    ):
        if idx == 0:
            batch_size = images.shape[0]
            batch_counter = batch_size * epoch

        # move tensors to the right device
        images, targets = images.to(device), targets.to(device)
        images = (images - mean) / std

        # forward pass
        predictions = model(images)
        loss = loss_func(predictions, targets)

        # logging
        batch_loss = loss.detach().item() / batch_size
        batch_losses.append(batch_loss)
        batch_logger.info(
            f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {batch_loss:>6.4f}"
        )
        if writer:
            writer.add_scalar("eval batch loss", batch_loss, batch_counter)
            batch_counter += 1
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_logger.info(f"epoch: {epoch:>2} | loss: {epoch_loss:>6.4f}")
    if writer:
        writer.add_scalar("eval epoch loss", epoch_loss, epoch)

    return epoch_loss

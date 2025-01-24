from tqdm import tqdm


def evaluate_model(
    model, data_loader, loss_func, device, epoch, batch_logger, epoch_logger, writer
):

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

        # forward pass
        predictions = model(images)
        loss = loss_func(predictions, targets)

        # logging
        batch_loss = loss.detach().item() / batch_size
        batch_losses.append(batch_loss)
        batch_logger.info(
            f"epoch: {epoch:>2} | batch: {idx:>3} | loss: {batch_loss:>6.3f}"
        )
        if writer:
            writer.add_scalar("eval batch loss", batch_loss, batch_counter)
            batch_counter += 1
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_logger.info(f"epoch: {epoch:>2} | loss: {epoch_loss:>6.3f}")
    if writer:
        writer.add_scalar("eval epoch loss", epoch_loss, epoch)

import datetime
import os
import logging

LOGGING_PATH = "/home/masn/projects/yolo/logs"


def evaluate_model(model, data_loader, yolo_loss, device, epoch : int = 0):
    batch_losses = []
    model.eval()
    for idx, (images, targets) in enumerate(data_loader):  
        if idx == 0:
            batch_size = images.shape[0]
            
        # move tensors to the right device
        images, targets = images.to(device), targets.to(device)

        # forward pass
        predictions = model(images)
        loss = yolo_loss(predictions, targets)
        
        # logging
        batch_losses.append(loss.detach().item())
        #eval_logger.info(f'batch: {idx:>3} | loss: {loss.detach().item() / batch_size :>6.3f}')
    #epoch_logger.info(f'epoch: {epoch:>3} | loss: {sum(batch_losses) / len(batch_losses) :>6.3f}')
    #print(sum(batch_losses) / len(batch_losses))
   
   
import torch
import time
from tqdm import tqdm
import os
import wandb

def train_model(epochs, train_loader, num_batches_per_epoch, model, optimizer, lr_scheduler,
                validation_iter=None, device='cuda', model_save_path='./model_checkpoints/', 
                log_interval=100):
    """
    A function to train the model and return the average validation loss.
    :param epochs: Number of epochs to train.
    :param train_loader: DataLoader for training data.
    :param num_batches_per_epoch: Number of batches in each epoch.
    :param model: The neural network model to train.
    :param optimizer: Optimizer used for training.
    :param validation_iter: DataLoader for validation data, if any.
    :param device: Device to train on, 'cuda' or 'cpu'.
    :param model_save_path: Path to save model checkpoints.
    :param log_interval: Interval to log training progress.
    
    :return: Average validation loss over the validation dataset.
    """
    losses_t = []
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    for epoch in range(epochs):
        model.train()
        cumm_epoch_loss = 0.0

        with tqdm(train_loader, total=num_batches_per_epoch - 1) as it:
            for batch_no, data_entry in enumerate(it, start=1):
                optimizer.zero_grad()
                signals = data_entry['signals'].to(device)
                losses = model.log_prob(signals)
                cumm_epoch_loss += losses.item()

                avg_epoch_loss = cumm_epoch_loss / batch_no
                it.set_postfix({"epoch": f"{epoch + 1}/{epochs}", "avg_loss": avg_epoch_loss}, refresh=False)

                wandb.log({"train_loss": losses.item()})
                losses.backward()
                optimizer.step()
                
                

                if num_batches_per_epoch == batch_no:
                    break

        losses_t.append(avg_epoch_loss)
        if (epoch + 1) % log_interval == 0:
            model_checkpoint_path = os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_checkpoint_path)
            print(f'Model saved to {model_checkpoint_path}')

    # Validation loop
    if validation_iter is not None:
        model.eval()
        cumm_epoch_loss_val = 0.0
        with tqdm(validation_iter, total=num_batches_per_epoch - 1, colour="green") as it:
            for batch_no, data_entry in enumerate(it, start=1):
                signals = data_entry['signals'].to(device)
                with torch.no_grad():
                    losses = model.log_prob(signals)

                cumm_epoch_loss_val += losses.item()
                avg_epoch_loss_val = cumm_epoch_loss_val / batch_no

                it.set_postfix({"epoch": f"{epoch + 1}/{epochs}", "avg_val_loss": avg_epoch_loss_val}, refresh=False)

        return avg_epoch_loss_val
    else:
        lr_scheduler.step()
        #lr_scheduler.step(sum(losses_t) / len(losses_t))
        return sum(losses_t) / len(losses_t)

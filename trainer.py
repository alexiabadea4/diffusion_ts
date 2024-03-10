import time
from typing import List, Optional, Union

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from diff_ts.gaussian_diffusion import GaussianDiffusion
from diff_ts.epsilon_theta import EpsilonTheta
from gluonts.core.component import validated
import wandb
import os

class Trainer:

    def __init__(
            self,
            epochs: int = 10,
            batch_size: int = 32,
            num_batches_per_epoch: int = 50,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-6,
            maximum_learning_rate: float = 1e-2,
            model_name : str = 'model',
            model_type : str ='torch',
            model_save_path : str = 'model_sav_path',
            input_size = [ 256],
            net = GaussianDiffusion(EpsilonTheta([256]), input_size = [256]),

            **kwargs,
    )->None:
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.model_name = model_name
        self.model_type = model_type
        self.model_save_path = model_save_path
        self.input_size = input_size
        self.net = net
        


    
    def __call__(
            self,
           
            train_iter: DataLoader,
 
    )->None:
        
        wandb.login()
        
        wandb.init(project="test_train_2")

        # Log hyperparameters and other configurations
        config = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'num_batches_per_epoch': self.num_batches_per_epoch,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'maximum_learning_rate': self.maximum_learning_rate,
        }
        wandb.config.update(config)

        optimizer = Adam(
            self.net.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr = self.maximum_learning_rate,
            steps_per_epoch = self.num_batches_per_epoch,
            epochs = self.epochs,
        ) 

        losses_t = []
        for epoch in range(self.epochs):
            tic = time.time()
            cumm_epoch_loss = 0.0

            with tqdm(train_iter, total=self.num_batches_per_epoch - 1) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    signals = data_entry['signals']
                    losses = self.net.log_prob(signals)
                    cumm_epoch_loss += losses.item()

                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix({"epoch": f"{epoch + 1}/{self.epochs}", "avg_loss": avg_epoch_loss}, refresh=False)

                    wandb.log({"train_loss": losses.item()})
                    losses.backward()
                    optimizer.step()
                
                

                    if self.num_batches_per_epoch == batch_no:
                        break
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gradient_norm": self.calculate_gradient_norm(self.net),
                "training_time_per_epoch": time.time() - tic,
            })
            losses_t.append(avg_epoch_loss)
        
        self.save_model_as_artifact(self.net)
            

    @staticmethod
    def calculate_gradient_norm(net):
        total_norm = 0.0
        for param in net.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def save_model_as_artifact(self, model):
        model.eval()  # Set the model to evaluation mode

        # Save the model state dictionary instead of using ONNX
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        model_path = os.path.join(self.model_save_path, f'{self.model_name}.pth')
        torch.save(model.state_dict(), model_path)

        # Create an artifact for logging to wandb
        artifact = wandb.Artifact(self.model_name, type=self.model_type)
        
        # Add metadata to the artifact
        artifact.metadata = {
            'format': 'pytorch_state_dict',
            'model_type': self.model_type,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'layers': [str(layer) for layer in model.children()],
        }
        
        # Add the model file to the artifact
        artifact.add_file(model_path)

        # Log the artifact to wandb
        wandb.log_artifact(artifact)


    
    def custom_collate_fn(self,batch):
        """
        Custom collate function to reshape data into [batch size, channels, size].
        """
        # Assuming your signals are originally in the shape [size]
        # and you want to add a single channel dimension
        signals = torch.stack([item['signals'] for item in batch]).unsqueeze(1)  # Adds a channel dimension
        gt = torch.stack([item['gt'] for item in batch])
        sc = torch.stack([item['sc'] for item in batch])
        
        return {'signals': signals, 'gt': gt, 'sc': sc}



        


    
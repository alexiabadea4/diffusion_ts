import torch as T
import numpy as np
from .DGdif import generate_balanced_dataset

class SynthSignalsDataset(T.utils.data.Dataset):
    """A PyTorch Dataset class for a balanced dataset of synthetic signals."""

    def __init__(self, num_samples_per_class, noise_level=0.0, device=None):
        """
        Initialize the dataset with balanced classes.
        
        Parameters:
        num_samples_per_class (int): Number of samples per class.
        noise_level (float): The noise level to be added to the signals.
        device (str, optional): The device to store the tensors on (e.g., 'cuda' or 'cpu').
        """
        x_tmp, y_tmp, signalclasses = generate_balanced_dataset(num_samples_per_class, noise_level)
        self.x_data = T.tensor(x_tmp, dtype=T.float32).to(device)
        self.y_data = T.tensor(y_tmp, dtype=T.long).to(device)
        self.signalclasses = T.tensor(signalclasses, dtype=T.long).to(device)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Parameters:
        idx (int): The index of the sample to retrieve.
        
        Returns:
        dict: A dictionary containing the signals, ground truth, and class.
        """
        signals = self.x_data[idx]
        gt = self.y_data[idx]
        sc = self.signalclasses[idx]
        return {'signals': signals, 'gt': gt, 'sc': sc}
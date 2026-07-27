# file for helper functions and visualizations

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

# for interactive desplay

from src.config import model_save_dir, latent_dimension_size
from src.model import VAE
from src.dataset import get_dataloaders

# Saves the model state dictionary to disk.
def save_model(model):
    model_save_dir.mkdir(parents=True, exist_ok=True)
    filename = "convolutional_variational_autoencoder.pth"
    save_path = model_save_dir / filename
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
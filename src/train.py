# file where the model is trained

import torch
import torch.nn as nn
import torch.nn.functioznal as F 

from src.dataset import get_dataloaders
from src.model import CVAE, VAE
from src.utils import save_model
import src.config as config



def vae_loss_function(reconstructed, original, mu, logvar, beta=config.beta_value):
    # Reconstruction Loss
    MSE = F.mse_loss(reconstructed, original, reduction='sum')
    
    # Kullback-Leibler Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = MSE + (beta * KLD)
    return total_loss, MSE, KLD

def train_model(model="VAE"):
        
    train_loader, val_loader = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

	# update when new models are added 
    if model == "VAE":
        model = VAE().to(device)
    else:
        model = CVAE().to(device)
    print(f"Model: {model}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)
    
    metrics = {
        'train_mse': [], 'train_kld': [],
        'val_mse': [], 'val_kld': []
    }

    for epoch in range(config.num_epochs):
        model.train() 
        
        running_train_mse = 0.0
        running_train_kld = 0.0
        
        for data, _ in train_loader:
            data = data.to(device) 
            
            reconstructed, mu, logvar = model(data)
            loss, mse, kld = vae_loss_function(reconstructed, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_mse += mse.item() / len(data)
            running_train_kld += kld.item() / len(data)
            
        metrics['train_mse'].append(running_train_mse / len(train_loader))
        metrics['train_kld'].append(running_train_kld / len(train_loader))
        
        model.eval()
        running_val_mse = 0.0
        running_val_kld = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                reconstructed, mu, logvar = model(data)
                
                loss, mse, kld = vae_loss_function(reconstructed, data, mu, logvar)
                running_val_mse += mse.item() / len(data)
                running_val_kld += kld.item() / len(data)
                
        metrics['val_mse'].append(running_val_mse / len(val_loader))
        metrics['val_kld'].append(running_val_kld / len(val_loader))
        
        print(f"Epoch {epoch} | Val MSE: {metrics['val_mse'][-1]:.4f} | Val KLD: {metrics['val_kld'][-1]:.4f}")

	
    save_model(model)
    return metrics

if __name__ == "__main__":
    train_model(model)
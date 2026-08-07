# file where the model is trained

from tabnanny import verbose

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from src.dataset import get_dataloaders
from src.model import CAE, VAE
from src.utils import save_model, evaluate_reconstruction_variance
from src.config import beta_value, Learning_rate, num_epochs, early_stopping_min_delta, early_stopping_patience, lambda_value, model_save_dir

def vae_loss_function(reconstructed, original, mu, logvar, beta=beta_value):
    # Reconstruction Loss
    mse = F.mse_loss(reconstructed, original, reduction="sum")

    # Kullback-Leibler Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = (lambda_value * mse) + (beta * kld)
    return total_loss, mse, kld


def train_model(data, model="VAE", beta=beta_value, verbose=True):
    from src.config import profile_features
    condition_index = [len(profile_features)]
    
    train_loader, val_loader, test_loader = get_dataloaders(data, condition_cols=condition_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training on device: {device}")

    # update when new models are added
    if model == "VAE":
        model = VAE().to(device)
    else:
        model = CAE().to(device)

    if verbose:
        print(f"Model: {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

    # Cosine Annealing Scheduler
    # T_max set to num_epochs to stretch the descent curve over the full training duration.
    # eta_min enforces a non-zero terminal state to prevent premature convergence into local optima.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=Learning_rate * 0.01)

    metrics = {
        "train_mse": [],
        "train_kld": [],
        "val_mse": [],
        "val_kld": [],
        "total_val_loss": [],
    }

    prev_gap = None
    patience_counter = 0
    anneal_epochs = num_epochs * 0.25

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()

        running_train_mse = 0.0
        running_train_kld = 0.0

        for data, _ in train_loader:
            data = data.to(device)

            # Flatten [batch_size, 60, 2] to [batch_size, 120]
            data_flat = data.view(data.size(0), -1)

            reconstructed, mu, logvar = model(data_flat)
            loss, mse, kld = vae_loss_function(reconstructed, data_flat, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_mse += mse.item() / len(data)
            running_train_kld += kld.item() / len(data)

        metrics["train_mse"].append(running_train_mse / len(train_loader))
        metrics["train_kld"].append(running_train_kld / len(train_loader))

        model.eval()
        running_val_mse = 0.0
        running_val_kld = 0.0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)

                # Flatten [batch_size, 60, 2] to [batch_size, 120]
                data_flat = data.view(data.size(0), -1)

                reconstructed, mu, logvar = model(data_flat)        

                loss, mse, kld = vae_loss_function(reconstructed, data_flat, mu, logvar, beta=beta)
                running_val_mse += mse.item() / len(data)
                running_val_kld += kld.item() / len(data)

        metrics["val_mse"].append(running_val_mse / len(val_loader))
        metrics["val_kld"].append(running_val_kld / len(val_loader))

        train_mse_loss = metrics["train_mse"][-1]
        val_mse_loss = metrics["val_mse"][-1]   
        train_kld_loss = metrics["train_kld"][-1]
        val_kld_loss = metrics["val_kld"][-1]

        total_train_loss = (lambda_value * train_mse_loss) + (beta * train_kld_loss)
        total_val_loss = (lambda_value * val_mse_loss) + (beta * val_kld_loss)
        
        metrics["total_val_loss"].append(total_val_loss)

        # Enforce early stopping delta threshold against total objective loss
        if total_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = total_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epoch_marker = "*"
            patience_counter = 0
        else:
            epoch_marker = " "
            patience_counter += 1

        gap_mse = abs(train_mse_loss - val_mse_loss)
        gap_kld = abs(train_kld_loss - val_kld_loss)
        current_lr = scheduler.get_last_lr()[0]

        if verbose:
            print(
                f"Epoch {epoch} {epoch_marker}| LR: {current_lr:.2e} | Train MSE: {train_mse_loss:.4f} | Train KLD: {train_kld_loss:.4f} | Gap: {gap_mse:.4f} | Total Train: {total_train_loss:.4f}"
            )
            print(
                f"      {' ' * ((len(str(epoch))) + len(f"  LR: {current_lr:.2e} "))}  | Val MSE: {val_mse_loss:.4f} | Val KLD: {val_kld_loss:.4f} | Gap: {gap_kld:.4f} | Total Val: {total_val_loss:.4f}"
            )

        scheduler.step()

        if patience_counter >= early_stopping_patience:
            if verbose:
                print("Early stopping triggered: Validation loss failed to improve by minimum delta.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored optimum matrix state mapping to Validation Loss: {best_val_loss:.4f}")

    save_model(model, verbose=verbose)

    return metrics


if __name__ == "__main__":
    train_model("VAE")
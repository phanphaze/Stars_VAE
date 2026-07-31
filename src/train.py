# file where the model is trained

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset import get_dataloaders
from src.model import CAE, VAE
from src.utils import save_model
import src.config as config


def vae_loss_function(reconstructed, original, mu, logvar, beta=config.beta_value, lambda_val=config.lambda_value):
    # Reconstruction Loss
    mse = F.mse_loss(reconstructed, original, reduction="sum")

    # Kullback-Leibler Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = (lambda_val * mse) + (beta * kld)
    return total_loss, mse, kld


def train_model(model="VAE"):
    train_loader, val_loader, test_loader = get_dataloaders(condition_cols=[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # update when new models are added
    if model == "VAE":
        model = VAE().to(device)
    else:
        model = CAE().to(device)
    print(f"Model: {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.Learning_rate)

    metrics = {
        "train_mse": [],
        "train_kld": [],
        "val_mse": [],
        "val_kld": [],
    }

    prev_gap = None
    patience_counter = 0

    for epoch in range(config.num_epochs):
        model.train()

        running_train_mse = 0.0
        running_train_kld = 0.0

        for data, _ in train_loader:
            data = data.to(device)

            # Flatten [batch_size, 60, 2] to [batch_size, 120]
            data_flat = data.view(data.size(0), -1)

            reconstructed, mu, logvar = model(data_flat)
            loss, mse, kld = vae_loss_function(reconstructed, data_flat, mu, logvar)

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

                loss, mse, kld = vae_loss_function(reconstructed, data_flat, mu, logvar)
                running_val_mse += mse.item() / len(data)
                running_val_kld += kld.item() / len(data)

        metrics["val_mse"].append(running_val_mse / len(val_loader))
        metrics["val_kld"].append(running_val_kld / len(val_loader))

        train_mse_loss = metrics["train_mse"][-1]
        val_mse_loss = metrics["val_mse"][-1]
        train_kld_loss = metrics["train_kld"][-1]
        val_kld_loss = metrics["val_kld"][-1]

        total_train_loss = train_kld_loss + train_mse_loss
        total_val_loss = val_kld_loss + val_mse_loss

        gap_mse = abs(train_mse_loss - val_mse_loss)
        gap_kld = abs(train_kld_loss - val_kld_loss)
        gap = gap_kld + gap_mse

        if prev_gap is not None and gap > prev_gap + config.early_stopping_min_delta:
            patience_counter += 1
        else:
            patience_counter = 0

        prev_gap = gap

        print(
            f"Epoch {epoch} | Train MSE: {train_mse_loss:.4f} | Train KLD: {train_kld_loss:.4f} | Gap: {gap_mse:.4f} | Total Train: {total_train_loss}"
        )
        print(
            f"      {" " * (len(str(epoch)))} | Val MSE: {val_mse_loss:.4f} | Val KLD: {val_kld_loss:.4f} | Gap: {gap_kld:.4f} | Total Val: {total_val_loss}"
        )
        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered: train/validation divergence is increasing.")
            break

    save_model(model)
    return metrics


if __name__ == "__main__":
    train_model("VAE")
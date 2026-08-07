import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import CAE, VAE
from src.utils import save_model, evaluate_reconstruction_variance
from src.config import (
    beta_value,
    Learning_rate,
    num_epochs,
    early_stopping_min_delta,
    early_stopping_patience,
    lambda_value,
    model_save_dir,
    num_profile_points,
    profile_features,
)

# New: weight for the physics (PDE) loss term. Tune this -- start small
# (e.g. 1e-3 to 1e-6) and increase once you confirm training is stable,
# since PDE residuals in cgs units can be very large before normalization.
pde_weight = 1e-4


def compute_pde_loss(reconstructed_flat, aux_data):
    batch_size = reconstructed_flat.size(0)

    expected_features = len(profile_features)
    if expected_features != 4:
        raise ValueError(
            f"compute_pde_loss assumes 4 features [mass, logT, logP, luminosity], "
            f"but profile_features config has {expected_features}: {profile_features}. "
            f"Update this function's indexing to match the real layout."
        )

    recon = reconstructed_flat.view(batch_size, num_profile_points, expected_features)

    # Predicted profiles from VAE
    mr = recon[:, :, 0]         # mass
    log_t = recon[:, :, 1]      # logT
    log_p = recon[:, :, 2]      # logP
    lr = recon[:, :, 3]         # luminosity

    t = torch.pow(10.0, log_t)
    p = torch.pow(10.0, log_p)

    # Safe clamping to prevent NaN values from extreme or negative predictions
    t = torch.clamp(t, min=1e-10, max=1e12)
    p = torch.clamp(p, min=1e-10, max=1e35)

    # Safely extract or estimate radius (r) and density (rho)
    if aux_data is not None and aux_data.numel() > 0 and aux_data.shape[-1] > 0:
        log_r = aux_data[:, :, 0]
        r = torch.pow(10.0, log_r)
    else:
        r = torch.linspace(0.1, 1.0, num_profile_points, device=reconstructed_flat.device).unsqueeze(0).expand(batch_size, -1) * torch.clamp(mr, min=0.01)

    if aux_data is not None and aux_data.numel() > 0 and aux_data.shape[-1] > 1:
        log_rho = aux_data[:, :, 1]
        rho = torch.pow(10.0, log_rho)
    else:
        dr_shell = torch.clamp(r[:, 1:] - r[:, :-1], min=1e-6)
        dm_shell = torch.clamp(mr[:, 1:] - mr[:, :-1], min=1e-6)
        rho_vals = dm_shell / (4 * np.pi * torch.pow(0.5 * (r[:, :-1] + r[:, 1:]), 2) * dr_shell)
        rho = torch.cat([rho_vals[:, :1], rho_vals], dim=1)

    # Clamp spatial geometry variables
    r = torch.clamp(r, min=1e-6)
    rho = torch.clamp(rho, min=1e-15, max=1e15)

    # Opacity and nuclear energy generation estimates.
    # NOTE: these are still placeholders (fixed electron-scattering opacity,
    # zero energy generation). Eq. d currently just penalizes dL/dm != 0
    # everywhere. Replace with predicted/tabulated kappa and eps if you want
    # the physics loss to actually constrain those effects.
    kappa = torch.ones_like(mr) * 0.4
    eps = torch.zeros_like(mr)

    # Constants (cgs)
    G = 6.6743e-8
    a = 7.5657e-15
    c_light = 2.9979e10

    # Spatial derivatives across the mass shells (size: batch_size, 59)
    dmr_step = torch.clamp(mr[:, 1:] - mr[:, :-1], min=1e-10)

    dp_dmr = (p[:, 1:] - p[:, :-1]) / dmr_step
    dr_dmr = (r[:, 1:] - r[:, :-1]) / dmr_step
    dt_dmr = (t[:, 1:] - t[:, :-1]) / dmr_step
    dlr_dmr = (lr[:, 1:] - lr[:, :-1]) / dmr_step

    # Midpoints for matching dimensions
    r_mid = 0.5 * (r[:, :-1] + r[:, 1:])
    mr_mid = 0.5 * (mr[:, :-1] + mr[:, 1:])
    rho_mid = 0.5 * (rho[:, :-1] + rho[:, 1:])
    t_mid = 0.5 * (t[:, :-1] + t[:, 1:])
    kappa_mid = 0.5 * (kappa[:, :-1] + kappa[:, 1:])
    lr_mid = 0.5 * (lr[:, :-1] + lr[:, 1:])
    eps_mid = 0.5 * (eps[:, :-1] + eps[:, 1:])

    # Residuals for Eqs 1.15a-d
    res_a = dp_dmr - (- (G * mr_mid) / (4 * np.pi * torch.pow(r_mid, 4)))
    res_b = dr_dmr - (1.0 / (4 * np.pi * torch.pow(r_mid, 2) * rho_mid))
    res_c = dt_dmr - (- (3 * kappa_mid * lr_mid) / (64 * (np.pi**2) * a * c_light * torch.pow(t_mid, 3) * torch.pow(r_mid, 4)))
    res_d = dlr_dmr - eps_mid

    # Guard against inf/nan sneaking into the graph before they poison
    # gradients elsewhere in the loss.
    residual_sq_sum = res_a**2 + res_b**2 + res_c**2 + res_d**2
    residual_sq_sum = torch.nan_to_num(residual_sq_sum, nan=0.0, posinf=1e10, neginf=1e10)

    # Mean over (batch, shells) instead of raw sum, so this term's scale is
    # independent of batch_size and num_profile_points.
    pde_loss = torch.mean(residual_sq_sum)
    return pde_loss


def vae_loss_function(reconstructed, original, mu, logvar, aux_data, beta=beta_value, pde_weight=pde_weight):
    # Reconstruction Loss
    mse = F.mse_loss(reconstructed, original, reduction="sum")

    # Kullback-Leibler Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    data_loss = (lambda_value * mse) + (beta * kld)

    # Physics loss -- now actually included in the backward pass.
    pde_loss = compute_pde_loss(reconstructed, aux_data)

    total_loss = data_loss + (pde_weight * pde_loss)

    return total_loss, mse, kld, pde_loss


def train_model(data, model="VAE", beta=beta_value, verbose=False):
    train_loader, val_loader, test_loader = get_dataloaders(data, condition_cols=[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training on device: {device}")

    if model == "VAE":
        model = VAE().to(device)
    else:
        model = CAE().to(device)

    if verbose:
        print(f"Model: {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=Learning_rate * 0.01)

    metrics = {
        "train_mse": [],
        "train_kld": [],
        "train_pde": [],
        "val_mse": [],
        "val_kld": [],
        "val_pde": [],
        "total_val_loss": [],
    }

    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in tqdm(range(num_epochs)):
        model.train()

        running_train_mse = 0.0
        running_train_kld = 0.0
        running_train_pde = 0.0

        for data, _, aux_data in train_loader:
            data = data.to(device)
            aux_data = aux_data.to(device)
            optimizer.zero_grad()
            # Flatten [batch_size, 60, 4] to [batch_size, 240]
            data_flat = data.view(data.size(0), -1)

            reconstructed, mu, logvar = model(data_flat)
            loss, mse, kld, pde = vae_loss_function(reconstructed, data_flat, mu, logvar, aux_data, beta=beta)

            loss.backward()
            optimizer.step()

            # Per-sample-normalized running sums, matching how val is tracked.
            running_train_mse += mse.item() / len(data)
            running_train_kld += kld.item() / len(data)
            running_train_pde += pde.item()

        # Proper epoch averages over all batches, not just the last batch.
        metrics["train_mse"].append(running_train_mse / len(train_loader))
        metrics["train_kld"].append(running_train_kld / len(train_loader))
        metrics["train_pde"].append(running_train_pde / len(train_loader))

        model.eval()
        running_val_mse = 0.0
        running_val_kld = 0.0
        running_val_pde = 0.0

        with torch.no_grad():
            for data, _, aux_data in val_loader:
                data = data.to(device)
                aux_data = aux_data.to(device)

                # Flatten [batch_size, 60, 4] to [batch_size, 240]
                data_flat = data.view(data.size(0), -1)

                reconstructed, mu, logvar = model(data_flat)

                loss, mse, kld, pde = vae_loss_function(reconstructed, data_flat, mu, logvar, aux_data, beta=beta)
                running_val_mse += mse.item() / len(data)
                running_val_kld += kld.item() / len(data)
                running_val_pde += pde.item()

        metrics["val_mse"].append(running_val_mse / len(val_loader))
        metrics["val_kld"].append(running_val_kld / len(val_loader))
        metrics["val_pde"].append(running_val_pde / len(val_loader))

        train_mse_loss = metrics["train_mse"][-1]
        val_mse_loss = metrics["val_mse"][-1]
        train_kld_loss = metrics["train_kld"][-1]
        val_kld_loss = metrics["val_kld"][-1]
        train_pde_loss = metrics["train_pde"][-1]
        val_pde_loss = metrics["val_pde"][-1]

        total_train_loss = (lambda_value * train_mse_loss) + (beta * train_kld_loss) + (pde_weight * train_pde_loss)
        total_val_loss = (lambda_value * val_mse_loss) + (beta * val_kld_loss) + (pde_weight * val_pde_loss)

        metrics["total_val_loss"].append(total_val_loss)

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
                f"Epoch {epoch} {epoch_marker}| LR: {current_lr:.2e} | Train MSE: {train_mse_loss:.4f} | Train KLD: {train_kld_loss:.4f} | Train PDE: {train_pde_loss:.4e} | Gap: {gap_mse:.4f} | Total Train: {total_train_loss:.4f}"
            )
            print(
                f"Validation               | Val MSE: {val_mse_loss:.4f} | Val KLD: {val_kld_loss:.4f} | Val PDE: {val_pde_loss:.4e} | Gap: {gap_kld:.4f} | Total Val: {total_val_loss:.4f}"
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

    evaluate_reconstruction_variance(
        model=model,
        dataloader=val_loader,
        device=device
    )

    return metrics


if __name__ == "__main__":
    metrics = train_model(data=None)
    np.savez("../metrics.npz", **metrics)
# file for helper functions and visualizations

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import numpy as np
import torch

from src.preprocessing import process_simulation

# for interactive desplay

from src.config import model_save_dir, profile_features
from src.model import VAE
from src.dataset import get_dataloaders

# Saves the model state dictionary to disk.
def save_model(model):
    model_save_dir.mkdir(parents=True, exist_ok=True)
    filename = "convolutional_variational_autoencoder.pth"
    save_path = model_save_dir / filename
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


def varify_rdp(df, processed_df):
    # Extract the profiles directly from the RAW dataframe to keep the scale consistent
    raw_splitting_features, raw_profiles = process_simulation(df)

    # Select arbitrary profile index
    profile_idx = 0
    target_split_val_raw = raw_splitting_features[profile_idx]

    # Get the original, unscaled coordinates
    original_profile = raw_profiles[profile_idx][profile_features].to_numpy()

    # Isolate matching reduced profile from the final output matrix
    # We use np.isclose instead of '==' for safer float comparisons
    mask = np.isclose(processed_df[:, -1], target_split_val_raw)
    reduced_profile = processed_df[mask][:, :-1]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(original_profile[:, 0], original_profile[:, 1], label='Original', color='blue', alpha=0.4, marker='.')

    # Quick safety check to ensure points were found before plotting
    if len(reduced_profile) > 0:
        plt.plot(reduced_profile[:, 0], reduced_profile[:, 1], label='RDP Reduced', color='red', marker='x', linestyle='--')
    else:
        print(f"Warning: No reduced points found for age {target_split_val_raw}")

    plt.title(f"RDP Reduction Verification (Age: {target_split_val_raw:.2e})", fontsize=20)
    plt.xlabel(profile_features[0], fontsize=14)
    plt.ylabel(profile_features[1], fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_curve(metrics, hyperparams=None, figsize=(15, 5), show_hyperparams=True):
    """
    Plot training/validation MSE and KLD curves on a log scale.
    metrics should include:
        - train_mse
        - val_mse
        - train_kld
        - val_kld
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    if hyperparams is None:
        try:
            from src.config import beta_value, latent_dimension_size, batch_size, Learning_rate
            hyperparams = {
                "beta": beta_value,
                "latent_dim": latent_dimension_size,
                "learning rate": Learning_rate,
                "batch_size": batch_size,
            }
        except Exception:
            hyperparams = {}

    epochs = range(len(metrics["train_mse"]))

    train_mse = np.asarray(metrics["train_mse"], dtype=float)
    val_mse = np.asarray(metrics["val_mse"], dtype=float)
    train_kld = np.asarray(metrics["train_kld"], dtype=float)
    val_kld = np.asarray(metrics["val_kld"], dtype=float)

    eps = 1e-12
    train_mse = np.clip(train_mse, eps, None)
    val_mse = np.clip(val_mse, eps, None)
    train_kld = np.clip(train_kld, eps, None)
    val_kld = np.clip(val_kld, eps, None)

    def set_sparse_log_ticks(ax, values):
        vmin = np.min(values)
        vmax = np.max(values)
        ticks = []
        for exp in range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1):
            for base in [1, 2, 5]:
                val = base * 10**exp
                if vmin <= val <= vmax:
                    ticks.append(val)
        if not ticks:
            ticks = [10**int(np.floor(np.log10(vmin))), 10**int(np.ceil(np.log10(vmax)))]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, train_mse, label="Train MSE", linewidth=2.2)
    ax1.plot(epochs, val_mse, label="Validation MSE", linewidth=2.2)
    ax1.set_title("Reconstruction Loss (MSE)")
    ax1.set_xlabel("Epoch", fontsize=15)
    ax1.set_ylabel("MSE", fontsize=15)
    ax1.set_yscale("log")
    set_sparse_log_ticks(ax1, np.concatenate([train_mse, val_mse]))
    ax1.legend(frameon=True)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    ax2.plot(epochs, train_kld, label="Train KLD", linewidth=2.2)
    ax2.plot(epochs, val_kld, label="Validation KLD", linewidth=2.2)
    ax2.set_title("Latent Divergence (KLD)")
    ax2.set_xlabel("Epoch", fontsize=15)
    ax2.set_ylabel("KLD", fontsize=15)
    ax2.set_yscale("log")
    set_sparse_log_ticks(ax2, np.concatenate([train_kld, val_kld]))
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    if show_hyperparams and hyperparams:
        param_text = "\n".join(f"{k}: {v}" for k, v in hyperparams.items())
        fig.subplots_adjust(bottom=0.28, wspace=0.25)
        fig.text(
            0.5, 0.02,
            f"Hyperparameters\n{param_text}",
            ha="center",
            va="bottom",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9
            ),
        )
    else:
        fig.tight_layout()

    return fig, (ax1, ax2)
# file for helper functions and visualizations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
import unittest
import pandas as pd
from unittest.mock import patch

from src.preprocessing import process_simulation, get_max_perpendicular_distance, iterative_rdp_max_heap, rdp_preprocess

# for interactive desplay

from src.config import latent_dimension_size, num_profile_points, model_save_dir, profile_features
from src.model import VAE
from src.dataset import get_dataloaders

# Saves the model state dictionary to disk.
def save_model(model, path="model.pth", verbose=True):
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"Model successfully saved to {path}")

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


def plot_loss_curve(metrics, log_scale=True, show_hyperparams=True, hyperparams=None, figsize=(15, 5)):
    """
    Plot training/validation MSE and KLD curves.
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
            from src.config import beta_value, latent_dimension_size, batch_size, Learning_rate, lambda_value
            hyperparams = {
                "beta": beta_value,
                "latent_dim": latent_dimension_size,
                "learning rate": Learning_rate,
                "batch_size": batch_size,
                "lambda_val": lambda_value
            }
        except Exception:
            hyperparams = {}

    epochs = range(len(metrics["train_mse"]))

    train_mse = np.asarray(metrics["train_mse"], dtype=float)
    val_mse = np.asarray(metrics["val_mse"], dtype=float)
    train_kld = np.asarray(metrics["train_kld"], dtype=float)
    val_kld = np.asarray(metrics["val_kld"], dtype=float)

    if log_scale:
        eps = 1e-12
        train_mse = np.clip(train_mse, eps, None)
        val_mse = np.clip(val_mse, eps, None)
        train_kld = np.clip(train_kld, eps, None)
        val_kld = np.clip(val_kld, eps, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # MSE Plot
    ax1.plot(epochs, train_mse, label="Train MSE", linewidth=2.2)
    ax1.plot(epochs, val_mse, label="Validation MSE", linewidth=2.2)
    ax1.set_title("Reconstruction Loss (MSE)")
    ax1.set_xlabel("Epoch", fontsize=15)
    ax1.set_ylabel("MSE", fontsize=15)
    ax1.legend(frameon=True)
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    # KLD Plot
    ax2.plot(epochs, train_kld, label="Train KLD", linewidth=2.2)
    ax2.plot(epochs, val_kld, label="Validation KLD", linewidth=2.2)
    ax2.set_title("Latent Divergence (KLD)")
    ax2.set_xlabel("Epoch", fontsize=15)
    ax2.set_ylabel("KLD", fontsize=15)
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    # Dynamic Axis Configuration
    def configure_axis(ax):
        if log_scale:
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            
            ratio = ymax / ymin
            
            if ratio < 5:
                # Extremely narrow range: override with linear locators to prevent tick starvation
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            elif ratio < 100:
                # Medium range: restrict subdivisions to 1, 2, and 5 to prevent text collision
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            elif ratio < 1000:
                # Wide range: restrict subdivisions to 1 and 5
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 5.0)))
            else:
                # Very wide range: utilize only major powers of 10
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            
            # Format major ticks to standard notation and suppress minor labels
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.set_yscale("linear")

    configure_axis(ax1)
    configure_axis(ax2)

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

def generate_stellar_profiles(model_path, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize architecture and load trained weights
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Sample from standard normal distribution N(0, I)
    z = torch.randn(num_samples, latent_dimension_size).to(device)

    with torch.no_grad():
        # Pass latent vectors through the decoder
        generated_flat = model.decode(z) 
        
    # Reshape from flattened tensor to original profile dimensions: [samples, 60, 2]
    num_features = len(profile_features)
    generated_profiles = generated_flat.view(num_samples, num_profile_points, num_features)
    
    return generated_profiles.cpu().numpy()

def evaluate_reconstruction_variance(model, dataloader, device):
    """Evaluates and prints the reconstruction variance over a dataloader."""
    model.eval()
    mse_list = []
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch unpacking structures
            if len(batch) == 3:
                data, _, _ = batch
            else:
                data, _ = batch
                
            data = data.to(device)
            data_flat = data.view(data.size(0), -1)
            reconstructed, _, _ = model(data_flat)
            mse = F.mse_loss(reconstructed, data_flat, reduction="mean")
            mse_list.append(mse.item())
            
    avg_mse = sum(mse_list) / len(mse_list) if mse_list else 0.0
    print(f"Validation Reconstruction Variance (MSE): {avg_mse:.6f}")
    return avg_mse


'''Test class for RDP Algorithm'''

# class TestRDPAlgorithm(unittest.TestCase):
    
#     def test_get_max_perpendicular_distance(self):
#         points = np.array([
#             [0.0, 0.0],
#             [1.0, 2.0],
#             [2.0, 0.5],
#             [3.0, 0.0]
#         ])
        
#         dist, split_idx = get_max_perpendicular_distance(points, 0, 3)
#         self.assertAlmostEqual(dist, 2.0, places=5)
#         self.assertEqual(split_idx, 1)

#     def test_get_max_perpendicular_distance_collinear(self):
#         points = np.array([
#             [0.0, 0.0],
#             [1.0, 1.0],
#             [2.0, 2.0],
#             [3.0, 3.0]
#         ])
        
#         dist, split_idx = get_max_perpendicular_distance(points, 0, 3)
#         self.assertAlmostEqual(dist, 0.0)

#     def test_iterative_rdp_max_heap(self):
#         points = np.array([
#             [0.0, 0.0],
#             [1.0, 0.1],
#             [2.0, 5.0],
#             [3.0, 0.2],
#             [4.0, 0.0]
#         ])
#         original_indices = np.array([100, 101, 102, 103, 104])
        
#         target_points = 3
#         rdp_indices = iterative_rdp_max_heap(points, original_indices, target_points)
        
#         np.testing.assert_array_equal(rdp_indices, [100, 102, 104])

#     def test_iterative_rdp_target_exceeds_points(self):
#         points = np.array([[0,0], [1,1], [2,0]])
#         original_indices = np.array([0, 1, 2])
#         rdp_indices = iterative_rdp_max_heap(points, original_indices, 10)
        
#         np.testing.assert_array_equal(rdp_indices, [0, 1, 2])

#     @patch('src.preprocessing.process_simulation')
#     @patch('src.preprocessing.fit_preprocess_scalers')
#     def test_rdp_preprocess_integration(self, mock_fit, mock_process):
#         np.random.seed(42)
#         raw_df = pd.DataFrame({
#             'feature1': np.random.rand(10),
#             'feature2': np.random.rand(10),
#             'zone': [1.0]*5 + [2.0]*5
#         }, index=[10, 11, 12, 13, 14, 20, 21, 22, 23, 24])

#         mock_fit.return_value = (raw_df, {})
        
#         profile_1 = raw_df.iloc[0:5]
#         profile_2 = raw_df.iloc[5:10]
#         mock_process.return_value = ([1.0, 2.0], [profile_1, profile_2])

#         profile_features = ['feature1', 'feature2']
#         split_feature = 'zone'
#         target_points = 3
        
#         output_npy = rdp_preprocess(
#             raw_df=raw_df,
#             split_feature=split_feature,
#             profile_features=profile_features,
#             num_profile_points=target_points
#         )
        
#         self.assertEqual(output_npy.shape, (6, 3))
#         self.assertTrue(np.all(output_npy[:, 2] == np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])))

# unittest.main(argv=['first-arg-is-ignored'], exit=False)
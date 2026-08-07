# file for helper functions and visualizations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F

import numpy as np
import torch
import unittest
import pandas as pd
from unittest.mock import patch

from src.preprocessing import process_simulation, get_max_perpendicular_distance, iterative_rdp_max_heap, rdp_preprocess
from src.config import latent_dimension_size, num_profile_points, model_save_dir, profile_features
from src.model import VAE
from src.dataset import get_dataloaders

# Saves the model state dictionary to disk.
def save_model(model, verbose=True):
    model_save_dir.mkdir(parents=True, exist_ok=True)
    filename = "variational_autoencoder.pth"
    save_path = model_save_dir / filename
    
    torch.save(model.state_dict(), save_path)
    if verbose:
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
    mask = np.isclose(processed_df[:, -1], target_split_val_raw)
    reduced_profile = processed_df[mask][:, :-1]

    num_feats = len(profile_features)
    num_subplots = max(1, num_feats - 1)
    
    # Establish dynamic figure scaling based on feature count
    fig, axes = plt.subplots(1, num_subplots, figsize=(7 * num_subplots, 6))
    
    # Enforce iterable array architecture for single-subplot generation
    if num_subplots == 1:
        axes = [axes]

    for i in range(1, num_feats):
        ax = axes[i - 1]
        
        ax.plot(original_profile[:, 0], original_profile[:, i], label='Original', color='blue', alpha=0.4, marker='.')
        
        if len(reduced_profile) > 0:
            ax.plot(reduced_profile[:, 0], reduced_profile[:, i], label='RDP Reduced', color='red', marker='x', linestyle='--')
        else:
            if i == 1:
                print(f"Warning: No reduced points found for age {target_split_val_raw}")
                
        ax.set_title(f"{profile_features[i]} vs {profile_features[0]}", fontsize=16)
        ax.set_xlabel(profile_features[0], fontsize=14)
        ax.set_ylabel(profile_features[i], fontsize=14)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(f"RDP Reduction Verification (Age: {target_split_val_raw:.2e})", fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_loss_curve(metrics, log_scale=True, figsize=(16, 6)):
    """
    Plot training/validation curves with a vertical indicator for the optimal saved epoch.
    Optimized for physical poster presentation. Hyperparameter overlays removed for compliance 
    with academic poster standards.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # Enforce strict poster-grade typographic scale
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "axes.linewidth": 2,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
    })

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

    # Professional color palette
    train_color = "#1f77b4"
    val_color = "#d62728"
    optimum_color = "#2ca02c"

    # Isolate optimum point mapping directly to the saved state matrix
    best_epoch = None
    if "total_val_loss" in metrics:
        best_epoch = np.argmin(metrics["total_val_loss"])

    # MSE Plot - Train uses a thicker line and lower zorder; Val uses a thinner line and higher zorder
    ax1.plot(epochs, train_mse, label="Train MSE", color=train_color, linewidth=4, alpha=1, zorder=2)
    ax1.plot(epochs, val_mse, label="Validation MSE", color=val_color, linewidth=2, alpha=0.5, zorder=3)
    ax1.set_title("Reconstruction Loss (MSE)", pad=15)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")

    # KLD Plot
    ax2.plot(epochs, train_kld, label="Train KLD", color=train_color, linewidth=4, alpha=1, zorder=2)
    ax2.plot(epochs, val_kld, label="Validation KLD", color=val_color, linewidth=2, alpha=0.5, zorder=3)
    ax2.set_title("Latent Divergence (KLD)", pad=15)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("KLD")

    # Clean axes and apply target overlay
    for ax in (ax1, ax2):
        if best_epoch is not None:
            ax.axvline(x=best_epoch, color=optimum_color, linestyle="--", linewidth=2.5, alpha=1.0, zorder=1, label=f"Optimum (Epoch {best_epoch})")
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which="major", linestyle="-", alpha=0.15, color='black')
        
        # Deduplicate redundant legend labels and anchor position
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), 
            by_label.keys(), 
            loc="upper right", 
            frameon=True, 
            framealpha=0.9, 
            edgecolor="0.8"
        )

    # Dynamic Axis Configuration
    def configure_axis(ax):
        if log_scale:
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            
            ratio = ymax / ymin
            
            if ratio < 5:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            elif ratio < 100:
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            elif ratio < 1000:
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 5.0)))
            else:
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.set_yscale("linear")

    configure_axis(ax1)
    configure_axis(ax2)

    fig.tight_layout()
    return fig, (ax1, ax2)

def _get_default_model_and_loader(data, split="test"):
    """
    Automatically instantiates the VAE architecture, loads terminal matrix states,
    and isolates the requested data partition.
    """
    from src.model import VAE
    from src.dataset import get_dataloaders
    from src.config import profile_features, model_save_dir
    import torch
    
    features = len(profile_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_save_dir / "variational_autoencoder.pth", map_location=device))
    model.eval()

    train_loader, val_loader, test_loader = get_dataloaders(data=data, condition_cols=[features])
    
    if split == "test":
        loader = test_loader
    elif split == "val":
        loader = val_loader
    else:
        loader = train_loader
        
    return model, loader, device


def plot_profile_reconstruction(
    data,
    scalers, 
    title="Test Set Profile Reconstruction",
    model=None,
    dataloader=None,
    split="test"
):
    """
    Executes a deterministic reconstruction of a physical profile and renders a comparative visualization.
    Scales dynamically to n-dimensional feature configurations via subplot arrays.
    """
    from src.config import profile_features, num_profile_points
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import torch
    
    features = len(profile_features)

    if model is None or dataloader is None:
        d_model, d_loader, _ = _get_default_model_and_loader(data, split)
        model = model or d_model
        dataloader = dataloader or d_loader

    device = next(model.parameters()).device
    model.eval()
    
    batch = next(iter(dataloader))
    real_profile_scaled = batch[0][0].numpy()
    
    real_profile_tensor = torch.from_numpy(real_profile_scaled).float().view(1, -1).to(device)
    
    with torch.no_grad():
        reconstructed_tensor, _, _ = model(real_profile_tensor)
        
    synthetic_profile_scaled = reconstructed_tensor.view(num_profile_points, features).cpu().numpy()

    real_profile_unscaled = np.zeros_like(real_profile_scaled)
    synthetic_profile_unscaled = np.zeros_like(synthetic_profile_scaled)

    for i, feature_name in enumerate(profile_features):
        real_profile_unscaled[:, i] = scalers[feature_name].inverse_transform(
            pd.DataFrame(real_profile_scaled[:, i], columns=[feature_name])
        ).flatten()
        
        synthetic_profile_unscaled[:, i] = scalers[feature_name].inverse_transform(
            pd.DataFrame(synthetic_profile_scaled[:, i], columns=[feature_name])
        ).flatten()

    num_subplots = max(1, features - 1)
    fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6))
    
    # Enforce iterable array architecture for single-subplot generation
    if num_subplots == 1:
        axes = [axes]
    elif hasattr(axes, "flatten"):
        axes = axes.flatten()
        
    feature_x = profile_features[0]
    
    for i in range(1, features):
        feature_y = profile_features[i]
        ax = axes[i - 1]
        
        ax.plot(real_profile_unscaled[:, 0], real_profile_unscaled[:, i], label="Real Profile", color="black", linewidth=2.5)
        ax.plot(synthetic_profile_unscaled[:, 0], synthetic_profile_unscaled[:, i], label="Synthetic Profile", color="#ff7f0e", linestyle="--", linewidth=2.5)
        
        ax.set_title(f"{feature_y} vs {feature_x}", fontsize=18)
        ax.set_xlabel(feature_x, fontsize=16) 
        ax.set_ylabel(feature_y, fontsize=16)
        
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle(title, fontsize=22, y=1.05)
    plt.tight_layout()
    plt.show()

    
# used for visualizing the latent space of the VAE model. Uses pca for interpretability and tsne for more accurate clustering.
def visualize_latent_space(
    reduction_method="tsne",
    sample_limit=5000,
    model_path=model_save_dir / "variational_autoencoder.pth"
):
    """
    Extracts and projects the VAE latent space into 2D for physical poster visualization.
    """
    from src.config import profile_features
    features = len(profile_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dynamic condition extraction
    _, val_loader, _ = get_dataloaders(condition_cols=[features])

    latent_vectors = []
    condition_values = []

    with torch.no_grad():
        for data, condition in val_loader:
            data = data.to(device)
            data_flat = data.view(data.size(0), -1)
            
            mu, _ = model.encode(data_flat)
            latent_vectors.append(mu.cpu().numpy())
            condition_values.append(condition.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    condition_values = np.concatenate(condition_values, axis=0).flatten()

    if len(latent_vectors) > sample_limit:
        indices = np.random.choice(len(latent_vectors), sample_limit, replace=False)
        latent_vectors = latent_vectors[indices]
        condition_values = condition_values[indices]

    if reduction_method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    else:
        reducer = PCA(n_components=2)
        
    latent_2d = reducer.fit_transform(latent_vectors)

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        latent_2d[:, 0], 
        latent_2d[:, 1], 
        c=condition_values, 
        cmap="viridis", 
        s=80, 
        alpha=0.85, 
        edgecolors="w", 
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Star Age (Normalized)", rotation=270, labelpad=30)

    ax.set_title(f"VAE Latent Space Distribution ({reduction_method.upper()})")
    ax.set_xlabel(f"{reduction_method.upper()} Component 1")
    ax.set_ylabel(f"{reduction_method.upper()} Component 2")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return fig, ax

def _plot_mse_distribution(mses):
    """
    Renders a histogram of the MSE distribution to diagnose normality and outlier structures.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(mses, bins=50, color='#1f77b4', edgecolor='black', alpha=0.75)
    
    median_mse = np.median(mses)
    mean_mse = np.mean(mses)
    
    ax.axvline(median_mse, color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Median: {median_mse:.4f}')
    ax.axvline(mean_mse, color='#d62728', linestyle='-', linewidth=2.5, label=f'Mean: {mean_mse:.4f}')
    
    ax.set_title("Profile Reconstruction Error Distribution (MSE)")
    ax.set_xlabel("Mean Squared Error")
    ax.set_ylabel("Frequency")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

def evaluate_reconstruction_variance(
    data, 
    model=None, 
    dataloader=None, 
    device=None, 
    num_examples=3, 
    split="test"
):
    """
    Computes per-sample MSE, sorts the distributions, and renders a diagnostic matrix 
    comparing the best, worst, and average reconstructions.
    """
    if model is None or dataloader is None or device is None:
        d_model, d_loader, d_device = _get_default_model_and_loader(data, split)
        model = model or d_model
        dataloader = dataloader or d_loader
        device = device or d_device

    model.eval()
    all_originals = []
    all_reconstructions = []
    all_mses = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            
            x_flat = x.view(x.size(0), -1)
            
            recon, _, _ = model(x_flat)
            
            mse_per_sample = F.mse_loss(recon, x_flat, reduction='none').mean(dim=1)
            
            all_originals.append(x_flat.cpu())
            all_reconstructions.append(recon.cpu())
            all_mses.append(mse_per_sample.cpu())

    originals = torch.cat(all_originals, dim=0).numpy()
    reconstructions = torch.cat(all_reconstructions, dim=0).numpy()
    mses = torch.cat(all_mses, dim=0).numpy()

    _plot_mse_distribution(mses)

    sorted_indices = np.argsort(mses)
    
    total_samples = len(sorted_indices)
    best_indices = sorted_indices[:num_examples]
    worst_indices = sorted_indices[-num_examples:]
    
    mid_point = total_samples // 2
    half_window = num_examples // 2
    avg_indices = sorted_indices[mid_point - half_window : mid_point - half_window + num_examples]

    categories = [
        ("Highest Fidelity (Lowest MSE)", best_indices),
        ("Median Fidelity (Average MSE)", avg_indices),
        ("Lowest Fidelity (Highest MSE)", worst_indices)
    ]

    _render_diagnostic_grid(originals, reconstructions, mses, categories, num_examples)

def _render_diagnostic_grid(originals, reconstructions, mses, categories, num_examples):
    """
    Renders a structural grid comparing original and reconstructed profiles.
    Reshapes flat vectors to physical profiles to expose structural deviation mechanics.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(len(categories), num_examples, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for row_idx, (title, indices) in enumerate(categories):
        for col_idx, idx in enumerate(indices):
            ax = axes[row_idx, col_idx]
            
            orig = originals[idx]
            recon = reconstructions[idx]
            error = mses[idx]

            # Reshape from [120] -> [60, 2] to isolate specific physical features
            orig_reshaped = orig.reshape(num_profile_points, len(profile_features))
            recon_reshaped = recon.reshape(num_profile_points, len(profile_features))

            x_axis = np.arange(num_profile_points)

            # Mass Plotting
            ax.plot(x_axis, orig_reshaped[:, 0], color='blue', linestyle='-', label=f'Orig {profile_features[0]}', alpha=0.8)
            ax.plot(x_axis, recon_reshaped[:, 0], color='cyan', linestyle='--', label=f'Recon {profile_features[0]}', alpha=0.8)
            
            # logT Plotting
            ax.plot(x_axis, orig_reshaped[:, 1], color='red', linestyle='-', label=f'Orig {profile_features[1]}', alpha=0.8)
            ax.plot(x_axis, recon_reshaped[:, 1], color='orange', linestyle='--', label=f'Recon {profile_features[1]}', alpha=0.8)

            if col_idx == 1:
                ax.set_title(f"{title}\nSample {idx} | MSE: {error:.4f}")
            else:
                ax.set_title(f"Sample {idx} | MSE: {error:.4f}")

            ax.set_xlabel("Profile Point Index")
            ax.set_ylabel("Normalized Value")
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.6)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.show()


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
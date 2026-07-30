# file for helper functions and visualizations

import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy

from src.preprocessing import process_simulation
from src.config import profile_features

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
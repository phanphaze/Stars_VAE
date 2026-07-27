# File for hyperparameters
from pathlib import Path

# Hyperparameters

# train attributes
Learning_rate = 1e-4
batch_size = 32
num_epochs = 50
train_test_split = 0.8
beta_value = 2 #weight of KL divergence

# model attributes
input_dimension_size = 2
output_dimension_size = 1
latent_dimension_size = 4


# Data paths
project_root = Path(__file__).resolve().parent.parent
raw_data_dir = project_root / "data" / "raw"
processed_data_path = project_root / "data" / "processed" / "processed_data.npy"
model_save_dir = project_root / "models"
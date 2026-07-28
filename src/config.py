# File for hyperparameters
from pathlib import Path

# Hyperparameters

# preprocessing attributes
rdp_epsilon = 0.05 #epsilon value for Ramer-Douglas-Peucker algorithm
target_features = ['mass', 'logT'] #features to be used for training

# train attributes
Learning_rate = 1e-4
batch_size = 32
num_epochs = 50
train_test_split = 0.8
beta_value = 2 #weight of KL divergence

# early stopping attributes
early_stopping_patience = 3 #number of epochs to wait for improvement before stopping
early_stopping_min_delta = 1e-4 #minimum change in loss to qualify as an improvement

# model attributes
input_dimension_size = 2
output_dimension_size = 2
latent_dimension_size = 4


# Data paths
project_root = Path(__file__).resolve().parent.parent
raw_data_dir = project_root / "data" / "raw"
processed_data_path = project_root / "data" / "processed" / "processed_data.npy"
model_save_dir = project_root / "models"
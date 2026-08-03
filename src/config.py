# File for hyperparameters
from pathlib import Path

# Hyperparameters

# rdp preprocessing attributes
profile_features = ['mass', 'logT'] #features to be used for training
split_feature = 'star_age'
num_profile_points = 60

# train attributes
Learning_rate = 1e-5
batch_size = 128
num_epochs = 5000
train_test_split = 0.8

# <1 for precise data reconstruction, 
# =1 for standard proibalistic training, 
# >1 for clean and separate feature learning)
beta_value = .3 #weight of KL divergence 
lambda_value = 1 #weight of MSE divergence


# early stopping attributes
early_stopping_patience = 30 #number of epochs to wait for improvement before stopping
early_stopping_min_delta = .01 #minimum change in loss to qualify as an improvement

# model attributes
input_dimension_size = num_profile_points * len(profile_features)
output_dimension_size = num_profile_points * len(profile_features)
hidden_dimension_1_size = 1024
latent_dimension_size = 8


# Data paths
project_root = Path(__file__).resolve().parent.parent
raw_data_path = project_root / "data" / "raw" / "sample_data.npy"
processed_data_path = project_root / "data" / "processed" / "processed_data.npy"
model_save_dir = project_root / "models"
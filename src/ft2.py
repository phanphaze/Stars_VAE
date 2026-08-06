from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.config import (
    profile_features,
    split_feature,
    num_profile_points,
    hidden_dimension_1_size,
    latent_dimension_size,
    Learning_rate,
    batch_size,
    num_epochs,
    beta_value,
    lambda_value,
    processed_data_path,
    model_save_dir,
)

# =========================================================
# CONFIG
# NOTE: config's input/output_dimension_size assumes reconstructing
# BOTH features. We only want temperature (logT) out, so we override
# output_dimension_size here instead of importing it from config.
# =========================================================
input_dimension = num_profile_points * len(profile_features)  # mass + logT -> 60*2 = 120
output_dimension = num_profile_points                          # logT only -> 60


# =========================================================
# Dataset
# =========================================================
class training(Dataset):
    """
    profs is a flat (N_rows, 3) array: columns are profile_features + [split_feature],
    i.e. [mass, logT, star_age]. Rows sharing the same star_age belong to one profile.
    This groups rows by star_age to reconstruct each individual profile.
    """
    def __init__(self, profs, profile_features, split_feature, num_profile_points):
        columns = profile_features + [split_feature]
        df = pd.DataFrame(profs, columns=columns)

        self.profiles = []
        for age, group in df.groupby(split_feature):
            if len(group) != num_profile_points:
                # skip any malformed/incomplete profile rather than crash
                continue
            self.profiles.append(group[profile_features].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx]         # shape (num_profile_points, 2) -> columns [mass, logT]
        mass_data = profile[:, 0]
        temp_data = profile[:, 1]

        x_input = np.concatenate([mass_data, temp_data])  # shape (2*num_profile_points,)
        x_output = temp_data                                # shape (num_profile_points,)

        return torch.from_numpy(x_input), torch.from_numpy(x_output)


# =========================================================
# Model
# =========================================================
class VAE(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_dimension=200, z_dimension=20):
        super().__init__()
        # Encoder
        self.in_2hid = nn.Linear(input_dimension, hidden_dimension)
        self.hid_2mu = nn.Linear(hidden_dimension, z_dimension)
        self.hid_2sigma = nn.Linear(hidden_dimension, z_dimension)
        # Decoder
        self.z_2hid = nn.Linear(z_dimension, hidden_dimension)
        self.hid_2out = nn.Linear(hidden_dimension, output_dimension)

    def encoder(self, x):
        a = nn.ReLU()(self.in_2hid(x))
        mu, sigma = self.hid_2mu(a), self.hid_2sigma(a)
        return mu, sigma

    def decoder(self, z):
        a = nn.ReLU()(self.z_2hid(z))
        return self.hid_2out(a)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z_reparameterized = mu + sigma * torch.randn_like(sigma)
        x_reconstructed = self.decoder(z_reparameterized)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    # ---- Sanity check with fake data ----
    x = torch.randn(4, input_dimension)
    model = VAE(input_dimension, output_dimension, hidden_dimension_1_size, latent_dimension_size)
    x_reconstructed, mu, sigma = model(x)
    print(x_reconstructed.shape)  # expect (4, 60)
    print(mu.shape)               # expect (4, 8)
    print(sigma.shape)            # expect (4, 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For a quick smoke test, override num_epochs temporarily, e.g.:
    # epochs = 2
    epochs = num_epochs

    profs = np.load(processed_data_path, allow_pickle=True)
    print(type(profs))
    print(profs.shape)
    print(profs[0])

    dataset = training(profs, profile_features, split_feature, num_profile_points)
    print(f"Loaded {len(dataset)} profiles")
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dimension, output_dimension, hidden_dimension_1_size, latent_dimension_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
    loss_function = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x_input, x_output) in loop:
            x_input = x_input.to(device)
            x_output = x_output.to(device)

            x_reconstructed, mu, sigma = model(x_input)
            reconstruction_loss = loss_function(x_reconstructed, x_output)
            kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            # weighted per config: lambda on reconstruction, beta on KL
            loss = lambda_value * reconstruction_loss + beta_value * kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    model = model.to("cpu")

    # =========================================================
    # Save the trained model
    # =========================================================
    model_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_dir / "vae_model.pt")
    print(f"Model saved to {model_save_dir / 'vae_model.pt'}")

    def reconstruction(profile_idx):
        profile = dataset.profiles[profile_idx]
        mass = profile[:, 0]
        temp = profile[:, 1]
        x_input = torch.from_numpy(np.concatenate([mass, temp])).unsqueeze(0)

        with torch.no_grad():
            mu, sigma = model.encoder(x_input)
            z = mu + sigma * torch.randn_like(sigma)
            out = model.decoder(z).squeeze(0).numpy()

        plt.figure()
        plt.plot(temp, label="true logT")
        plt.plot(out, label="reconstructed logT")
        plt.legend()
        plt.savefig(f"reconstruction_{profile_idx}.png")
        plt.close()

    n_profiles = len(dataset)
    for idx in [0, n_profiles // 2, n_profiles - 1]:
        reconstruction(idx)
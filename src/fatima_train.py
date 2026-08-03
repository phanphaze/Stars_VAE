from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset   
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim                         

mass_enclosed_key = "logR"
temp_key = "logT"
profs_shape = 977
 
class training(Dataset):
    def __init__(self, profs, mass_enclosed_key, temp_key):
        self.profs = profs
        self.mass_enclosed_key = mass_enclosed_key
        self.temp_key = temp_key
 
    def __len__(self):
        return len(self.profs)
 
    def __getitem__(self, idx):
        profile = self.profs[idx]
        mass_data = np.asarray(profile[self.mass_enclosed_key], dtype=np.float32)
        temp_data = np.asarray(profile[self.temp_key], dtype=np.float32)
 
        x_input = np.concatenate([mass_data, temp_data])  
        x_output = temp_data                              
 
        return torch.from_numpy(x_input), torch.from_numpy(x_output)
 
 

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
        # Linear output -- logT is not bounded in [0,1], so no sigmoid here.
        return self.hid_2out(a)
 
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z_reparameterized = mu + sigma * torch.randn_like(sigma)
        x_reconstructed = self.decoder(z_reparameterized)
        return x_reconstructed, mu, sigma
 
if __name__ == "__main__":
    input_dimension = 2 * profs_shape   # mass enclosed + temperature
    output_dimension = profs_shape      # temperature only
 
    # ---- Sanity check with fake data ----
    x = torch.randn(4, input_dimension)
    model = VAE(input_dimension, output_dimension)
    x_reconstructed, mu, sigma = model(x)
    print(x_reconstructed.shape) 
    print(mu.shape)               
    print(sigma.shape)            
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    batch_size = 32
    learning_rate = 1e-4
    hidden_dimension = 200
    z_dimension = 20
 
    # --- `profs` must exist before this point. Load your real data here: ---
    #profs = np.load("/work/nvme/bhvr/fatimasyed7/Stars_VAE/data/processed/processed_data.npy", allow_pickle=True)
    profs = np.load("/work/nvme/bhvr/fatimasyed7/Stars_VAE/data/processed/processed_data.npy", allow_pickle=True)
    print(type(profs))
    print(profs.shape)
    print(profs[0])
    dataset = training(profs, mass_enclosed_key, temp_key)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    profs = np.load("/work/nvme/bhvr/fatimasyed7/Stars_VAE/data/processed/processed_data.npy", allow_pickle=True)
    print(type(profs))
    print(profs.shape)
    print(profs[0])
 
    model = VAE(input_dimension, output_dimension, hidden_dimension, z_dimension).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss(reduction="sum")
 
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader))
        for i, (x_input, x_output) in loop:
            x_input = x_input.to(device)
            x_output = x_output.to(device)
 
            x_reconstructed, mu, sigma = model(x_input)
            reconstruction_loss = loss_function(x_reconstructed, x_output)
            kl_divergence = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + kl_divergence
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
 
    model = model.to("cpu")
 
    def reconstruction(snapshot_idx):
        mass = np.asarray(profs[snapshot_idx][mass_enclosed_key], dtype=np.float32)
        temp = np.asarray(profs[snapshot_idx][temp_key], dtype=np.float32)
        x_input = torch.from_numpy(np.concatenate([mass, temp])).unsqueeze(0)
 
        with torch.no_grad():
            mu, sigma = model.encoder(x_input)
            z = mu + sigma * torch.randn_like(sigma)
            out = model.decoder(z).squeeze(0).numpy()
 
        plt.figure()
        plt.plot(temp, label="true logT")
        plt.plot(out, label="reconstructed logT")
        plt.legend()
        plt.savefig(f"reconstruction_{snapshot_idx}.png")
        plt.close()
 
    for idx in [0, 100, 433]:
        reconstruction(idx)
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import os

try:
    import umap
    import matplotlib.pyplot as plt
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

@dataclass
class Config:
    data_dir: str = "./data"
    out_dir: str = "./out_ae"
    batch_size: int = 256
    epochs: int = 20
    lr: float = 2e-3
    latent_dim: int = 32
    recon_type: str = "mse"
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    sample_every: int = 1
    samples_per_epoch: int = 64

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 128, 7, 7)
        return self.deconv(h)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.logvar = nn.Parameter(-5.0*torch.ones(latent_dim))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        z_rand = torch.randn_like(z)
        std = torch.exp(0.5*self.logvar)
        return z, self.decode(z + std*z_rand)

def gaussian_nll_standard(z: torch.Tensor):
    """
    -log p(z) for z ~ N(0, I) up to constant.
    NLL = 0.5 * ||z||^2 + 0.5 * d * log(2π)
    Returns mean per-sample NLL.
    """
    d = z.size(1)
    quad = 0.5 * (z ** 2).sum(dim=1)  # [B]
    const = 0.5 * d * math.log(2 * math.pi)
    return (quad + const).mean()
           
def reconstruction_loss(x_pred, x):
    return F.mse_loss(x_pred, x, reduction="mean")

def set_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(cfg):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    return train_loader, test_ds

@torch.no_grad()
def plot_umap_latent(model, loader, cfg, epoch):
    if not HAS_UMAP:
        print("[UMAP] Skipping - umap-learn not installed")
        return
    model.eval()
    zs, ys = [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(cfg.device)
        z = model.encode(x)
        zs.append(z.cpu())
        ys.append(y)
        if i > 5:
            break
    Z = torch.cat(zs).numpy()
    Y = torch.cat(ys).numpy()
    reducer = umap.UMAP(n_components=2, random_state=cfg.seed)
    Z2 = reducer.fit_transform(Z)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=6)
    plt.colorbar(scatter)
    plt.title(f"AE Latent UMAP - epoch {epoch}")
    path = f"{cfg.out_dir}/umap_epoch_{epoch:03d}.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved UMAP plot to {path}")

def train(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    train_loader, _ = prepare_data(cfg)
    model = Autoencoder(cfg.latent_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for i, (x, _) in enumerate(train_loader):
            x = x.to(cfg.device)
            z, x_rec = model(x)

            # 1) Base recon loss
            loss_recon = reconstruction_loss(x_rec, x)

            # 2) Gentle Gaussian prior on codes (reduce from 10x to ~0.1–0.3)
            beta_prior = 0.002
            loss_prior = beta_prior * gaussian_nll_standard(z)

            # Total loss
            loss = loss_recon + loss_prior + 0.002*(model.logvar**2).mean()

            #loss = reconstruction_loss(x_rec, x) + gaussian_nll_standard(z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (i + 1) % cfg.log_every == 0:
                print(f"Epoch {epoch} Iter {i+1} Recon {loss.item():.4f}")
        if epoch % cfg.sample_every == 0:
            # 10x10 grid of RECONSTRUCTIONS (encode->decode of latest batch)
            grid_recon = vutils.make_grid(x_rec[:100], nrow=10, padding=2, normalize=False)
            vutils.save_image(grid_recon, f"{cfg.out_dir}/ae_recon_epoch_{epoch:03d}.png")

            # 10x10 grid of "SAMPLES" from a Gaussian prior (for comparison only)
            with torch.no_grad():
                z_rand = torch.randn(100, cfg.latent_dim, device=cfg.device)
                x_samp = model.decode(z_rand)
            grid_prior = vutils.make_grid(x_samp, nrow=10, padding=2, normalize=False)
            vutils.save_image(grid_prior, f"{cfg.out_dir}/ae_prior_epoch_{epoch:03d}.png")

            # Latent UMAP of encoded training images (if umap is installed)
            plot_umap_latent(model, train_loader, cfg, epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--data_dir", type=str, default=Config.data_dir)
    parser.add_argument("--out_dir", type=str, default=Config.out_dir)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--latent_dim", type=int, default=Config.latent_dim)
    args = parser.parse_args()
    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, latent_dim=args.latent_dim)
    if args.mode == "train":
        train(cfg)

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, output_padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
                               stride=2, padding=1,
                               output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DentalAutoencoder(nn.Module):

    def __init__(self, latent_channels: int = 128):
        super().__init__()

        #  Encoder
        self.encoder = nn.Sequential(
            ConvBlock(1,  32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, latent_channels, stride=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, 64),
            DeconvBlock(64, 32),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),                      
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)
        print(f"[Autoencoder] Saved → {path}")

    def load(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device)
        sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(sd)
        self.to(device)
        self.eval()
        print(f"[Autoencoder] Loaded ← {path}")


def build_autoencoder(latent_channels: int = 128,
                      device: str = "cpu") -> DentalAutoencoder:
    model = DentalAutoencoder(latent_channels=latent_channels)
    model = model.to(device)
    return model

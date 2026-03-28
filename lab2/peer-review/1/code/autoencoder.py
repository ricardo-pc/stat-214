import torch
import torch.nn as nn
import lightning as L


class Encoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, latent_dim),
        )
    def forward(self, x): return self.fc(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, 64 * 7 * 7), nn.LeakyReLU(0.1),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, in_channels, 3, padding=1),
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        return self.conv(x)


class Autoencoder(L.LightningModule):
    def __init__(self, patch_size=9, in_channels=8, latent_dim=8, optimizer_config=None):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(in_channels=in_channels, latent_dim=latent_dim)
        self.loss_fn = nn.MSELoss()
        self.optimizer_config = optimizer_config or {"lr": 1e-3, "weight_decay": 1e-5}

    def forward(self, x): return self.decoder(self.encoder(x))

    def embed(self, x): return self.encoder(x)

    def _shared_step(self, batch, stage):
        x = batch
        loss = self.loss_fn(self(x), x)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")

    def configure_optimizers(self):
        lr = self.optimizer_config.get("lr", 1e-3)
        wd = self.optimizer_config.get("weight_decay", 1e-5)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

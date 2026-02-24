"""MvGPRNet: Multi-view GPR 3D reconstruction network."""
from typing import Tuple

import torch
import torch.nn as nn


class Enhance2D(nn.Module):
    """2D enhancement block for encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.weight = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        combined = torch.cat((c1, c2), dim=1)
        out = self.weight(combined)
        if self.residual is not None:
            identity = self.residual(identity)
        return out + identity


class Enhance3D(nn.Module):
    """3D enhancement block for decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.weight = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        combined = torch.cat((c1, c2), dim=1)
        out = self.weight(combined)
        if self.residual is not None:
            identity = self.residual(identity)
        return out + identity


class MvGPRNetEncoder(nn.Module):
    """Encoder: multi-view 2D projections -> latent (mu, logvar)."""

    def __init__(self, n_views: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_views, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Enhance2D(64, 64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Enhance2D(128, 128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            Enhance2D(256, 256),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 16 * 16, 512)
        self.fc_mu = nn.Linear(512, 512)
        self.fc_logvar = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        h = self.flatten(x)
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class MvGPRNetDecoder(nn.Module):
    """Decoder: latent -> 3D volume."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 256 * 4 * 4 * 4)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Enhance3D(128, 128),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Enhance3D(64, 64),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Enhance3D(32, 32),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Enhance3D(16, 16),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 4, 4, 4)
        x = self.deconv1(h)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        volume_pred = self.deconv5(x)
        return volume_pred


class MvGPRNet(nn.Module):
    """Multi-view GPR 3D reconstruction."""

    def __init__(self, n_views: int = 20) -> None:
        super().__init__()
        self.encoder = MvGPRNetEncoder(n_views)
        self.decoder = MvGPRNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        else:
            z = mu
        volume_pred = self.decoder(z)
        return volume_pred

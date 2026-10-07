import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralBlock(nn.Module):
    """
    Building block for 1D Spectral extraction: Conv1D -> BatchNorm -> GELU -> MaxPool1D
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, pool_size: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=kernel_size // 2, 
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralEncoder(nn.Module):
    """
    AstroFoundation v4.0 1D Spectrographic Backbone.
    Ingests continuum-normalized rest-frame stellar spectra across a 4096 grid,
    extracts localized atomic absorption lines via 1D convolutions, preserves
    spatial wavelength band locality (16 regional bands) via Adaptive Pooling,
    and outputs a strictly contracted (BATCH, 256) float32 L2 unit-norm embedding.
    """
    def __init__(self, in_channels: int = 1, d_model: int = 256, num_spatial_bins: int = 16):
        super().__init__()
        self.num_spatial_bins = num_spatial_bins

        # 1. Hierarchical 1D Convolutional Stack (Downsamples 4096 grid -> 64 grid)
        # Progressively doubles filter channels (32 -> 64 -> 128)
        self.conv_stack = nn.Sequential(
            SpectralBlock(in_channels, 32, kernel_size=15, pool_size=4),  # 4096 -> 1024 (Wide atomic bands)
            SpectralBlock(32, 64, kernel_size=7, pool_size=4),            # 1024 -> 256  (Medium absorption lines)
            SpectralBlock(64, 128, kernel_size=5, pool_size=4)            # 256  -> 64   (Fine spectroscopic fingerprint)
        )

        # 2. Spatial-Preserving Adaptive Pooling across 16 wavelength regions
        # Preserves line wavelength positions (e.g. H-alpha vs O-III vs Ca H&K)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_spatial_bins)

        # 3. Projection Head for strict 256-dim output contract
        self.output_proj = nn.Sequential(
            nn.Linear(128 * num_spatial_bins, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Continuum-normalized 1D spectrum of shape (BATCH, 4096) or (BATCH, 1, 4096).
        Returns:
            normalized_embedding: Unit hypersphere vector of shape (BATCH, 256), ||v||_2 = 1.0.
        """
        # Step A: Ensure input is 3D -> (BATCH, 1, 4096)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Step B: Extract hierarchical spectroscopic features -> (BATCH, 128, 64)
        features = self.conv_stack(x)

        # Step C: Spatial-preserving pooling across 16 wavelength regions -> (BATCH, 128 * 16)
        pooled = self.adaptive_pool(features).flatten(1)

        # Step D: Project to system dimension -> (BATCH, 256)
        proj = self.output_proj(pooled)

        # Step E: Universal L2 Unit-Norm Contract (eps=1e-8 prevents zero-division NaN)
        normalized_embedding = F.normalize(proj, p=2, dim=-1, eps=1e-8)

        return normalized_embedding
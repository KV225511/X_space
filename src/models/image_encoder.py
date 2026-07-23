import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FITSEncoder(nn.Module):
    """
    AstroFoundation v4.0 Multi-Band Vision Backbone.
    Adapts a pretrained ResNet architecture to ingest 5-channel FITS astronomical
    images (u, g, r, i, z bands) and outputs a contracted (BATCH, 256) L2 unit-norm vector.
    """
    def __init__(self, in_channels: int = 5, d_model: int = 256):
        super().__init__()
        
        # 1. Load standard visual backbone (ResNet-34)
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # 2. Conv1 Adaptation: Replace 3-channel RGB ingest with C-channel astronomical ingest
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize new 5-channel weights by averaging pretrained RGB filters
        with torch.no_grad():
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            self.conv1.weight.data = rgb_mean.repeat(1, in_channels, 1, 1)
            
        # 3. Assemble feature extractor (strip original Conv1 and final classification FC layer)
        backbone.conv1 = self.conv1
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        # 4. Final Projection Head for strict 256-dim output contract
        num_features = backbone.fc.in_features
        self.output_proj = nn.Linear(num_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Arcsinh-stretched FITS image tensor of shape (BATCH, 5, 224, 224).
        Returns:
            normalized_embedding: Unit hypersphere vector of shape (BATCH, 256), ||v||_2 = 1.0.
        """
        # A. Extract spatial visual features -> (BATCH, 512, 1, 1)
        features = self.feature_extractor(x)
        
        # B. Flatten feature map -> (BATCH, 512)
        features = torch.flatten(features, 1)
        
        # C. Project to system dimension -> (BATCH, 256)
        proj = self.output_proj(features)
        
        # D. Universal L2 Unit-Norm Contract (eps=1e-8 prevents NaN on dead activations)
        normalized_embedding = F.normalize(proj, p=2, dim=-1, eps=1e-8)
        
        return normalized_embedding
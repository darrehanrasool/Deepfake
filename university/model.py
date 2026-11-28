"""ST2-SyncNet Multimodal Forensic Neural Network Architecture.

Tri-stream continuous spatial-wavelet decomposition, phase coherence,
and bi-directional temporal biological consistency modeling.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletDecomposition2D(nn.Module):
    """2D Haar Discrete Wavelet Transform (DWT) producing 12 sub-bands (4 per RGB channel)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        subbands = F.conv2d(x_reshaped, self.kernel, stride=2)
        _, _, sh, sw = subbands.shape
        return subbands.view(b, c * 4, sh, sw)


class SpectralPhaseCoherenceModule(nn.Module):
    """Exposes generator high-frequency upsampling grid residuals."""
    def __init__(self, in_channels: int = 12, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, subbands: torch.Tensor) -> torch.Tensor:
        feat = self.conv(subbands)
        return self.fc(torch.flatten(feat, 1))


class SpatialBoundaryAttentionModule(nn.Module):
    """Multi-scale convolutional backbone isolating face-swap boundary blending seams."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(x + self.res_block(x))
        x = self.head(x)
        return self.fc(torch.flatten(x, 1))


class TemporalBiologicalConsistencyModule(nn.Module):
    """Bi-directional GRU tracking optical flow and biological blinking continuity over frames."""
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        if hasattr(self.gru, 'flatten_parameters'):
            self.gru.flatten_parameters()
        gru_out, _ = self.gru(seq_feats)
        return self.fc(gru_out[:, -1, :])


class ST2SyncNet(nn.Module):
    """Full ST2-SyncNet Multimodal Deepfake Architecture with Cross-Attention Fusion."""
    def __init__(self, num_classes: int = 2, feature_dim: int = 128):
        super().__init__()
        self.spatial_encoder = SpatialBoundaryAttentionModule(out_dim=feature_dim)
        self.wavelet_transform = WaveletDecomposition2D()
        self.spectral_encoder = SpectralPhaseCoherenceModule(in_channels=12, out_dim=feature_dim)
        self.temporal_encoder = TemporalBiologicalConsistencyModule(feature_dim=feature_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.norm = nn.LayerNorm(feature_dim)

        # Classification Head with Anti-Overfitting Dropout (0.4)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        # 1. Spatial Boundary Stream
        f_spatial = self.spatial_encoder(x_spatial)

        # 2. 2D Wavelet Spectral Stream (12 Sub-bands)
        subbands = self.wavelet_transform(x_spatial)
        f_spectral = self.spectral_encoder(subbands)

        # 3. Cross-Attention: Spatial queries attend to Spectral keys/values
        q = f_spatial.unsqueeze(0)
        kv = torch.stack([f_spatial, f_spectral], dim=0)
        attn_out, _ = self.cross_attention(q, kv, kv)
        f_fused = self.norm(f_spatial + attn_out.squeeze(0))

        # 4. Temporal Biological Sequence Stream
        b, s, c, h, w = x_temporal.shape
        x_temp_flat = x_temporal.view(b * s, c, h, w)
        temp_spatial = self.spatial_encoder(x_temp_flat).view(b, s, -1)
        f_temporal = self.temporal_encoder(temp_spatial)

        # 5. Dual-Stream Concatenation & Classification
        f_combined = torch.cat([f_fused, f_temporal], dim=1)
        logits = self.classifier(f_combined)
        return logits

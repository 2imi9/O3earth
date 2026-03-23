"""
OlmoEarth Foundation Model for Renewable Energy Detection

Wrapper around Allen Institute's OlmoEarth vision transformer for:
- Installation detection (solar/wind/hydro)
- Capacity estimation from satellite imagery
- Change detection for construction monitoring
- Environmental compliance verification

Reference: https://allenai.org/blog/olmoearth-models
Model: Apache 2.0 License

Author: Zim (Millennium Fellowship Research)
Project: AI Earth Observation for Democratic Sustainable Energy Investment
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try loading real OlmoEarth pretrained weights
try:
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    from olmoearth_pretrain.data.constants import Modality as OlmoModality
    HAS_OLMOEARTH = True
except ImportError:
    HAS_OLMOEARTH = False


class TaskType(Enum):
    """Downstream tasks for renewable energy verification."""
    INSTALLATION_DETECTION = "installation_detection"
    ASSET_CLASSIFICATION = "asset_classification"
    CAPACITY_ESTIMATION = "capacity_estimation"
    CHANGE_DETECTION = "change_detection"
    COMPLIANCE_MONITORING = "compliance_monitoring"


class EnergySourceType(Enum):
    """Renewable energy source types."""
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    HYDRO = "hydro"
    NONE = "none"


@dataclass
class ModelConfig:
    """Configuration for OlmoEarth-based models."""

    # Base model
    model_name: str = "allenai/OLMoE"
    model_size: str = "base"  # "nano", "tiny", "base", "large"
    pretrained: bool = True
    use_real_backbone: bool = True  # Try real OlmoEarth weights

    # Input configuration
    input_channels: int = 12  # Sentinel-2 bands
    input_size: int = 224
    temporal_frames: int = 1  # For change detection, use >1

    # Architecture
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    patch_size: int = 16
    dropout: float = 0.1

    # Task-specific heads
    num_classes: int = 5  # For classification tasks

    # Training
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Freeze first N layers

    # Bands configuration (Sentinel-2 L2A)
    bands: List[str] = field(default_factory=lambda: [
        "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B11", "B12", "SCL", "CLD"
    ])


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings with spectral attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.proj = nn.Conv2d(
            config.input_channels,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Spectral attention for multi-band imagery
        self.spectral_attention = nn.Sequential(
            nn.Linear(config.input_channels, config.input_channels * 4),
            nn.GELU(),
            nn.Linear(config.input_channels * 4, config.input_channels),
            nn.Sigmoid()
        )
        
        # Learnable position embeddings
        num_patches = (config.input_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) multi-spectral image
        Returns:
            (B, N+1, D) patch embeddings with CLS token
        """
        B, C, H, W = x.shape
        
        # Apply spectral attention
        spectral_weights = self.spectral_attention(
            x.mean(dim=[2, 3])  # Global average per band
        )
        x = x * spectral_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Patch projection
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        return x


class TemporalEncoder(nn.Module):
    """Encode temporal information for change detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Temporal position encoding
        self.temporal_embed = nn.Parameter(
            torch.randn(1, config.temporal_frames, config.hidden_dim) * 0.02
        )
        
        # Cross-temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) temporal sequence of patch embeddings
        Returns:
            (B, N, D) temporally-aggregated embeddings
        """
        B, T, N, D = x.shape
        
        # Add temporal position encoding
        x = x + self.temporal_embed.unsqueeze(2)
        
        # Reshape for attention: treat patches as sequence
        x = x.view(B * N, T, D)
        
        # Self-attention across time
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm(x + attn_out)
        
        # Take last timestep or mean pool
        x = x.mean(dim=1)  # (B*N, D)
        x = x.view(B, N, D)
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class OlmoEarthBackbone(nn.Module):
    """
    OlmoEarth-style Vision Transformer backbone for Earth observation.
    
    Designed to match OlmoEarth architecture for potential weight loading
    while being trainable from scratch for renewable energy tasks.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Temporal encoder (optional)
        if config.temporal_frames > 1:
            self.temporal_encoder = TemporalEncoder(config)
        else:
            self.temporal_encoder = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_all_tokens: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W) for temporal
            return_all_tokens: If True, return all patch tokens
        Returns:
            CLS token embedding or (CLS, all tokens)
        """
        # Handle temporal input
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Process each frame
            temporal_embeds = []
            for t in range(T):
                temporal_embeds.append(self.patch_embed(x[:, t]))
            x = torch.stack(temporal_embeds, dim=1)  # (B, T, N, D)
            
            # Temporal aggregation
            if self.temporal_encoder:
                x = self.temporal_encoder(x)
            else:
                x = x.mean(dim=1)
        else:
            x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if return_all_tokens:
            return x[:, 0], x[:, 1:]  # CLS token, patch tokens
        return x[:, 0]  # CLS token only


class OlmoEarthRealBackbone(nn.Module):
    """Wrapper around real OlmoEarth pretrained encoder from allenai.

    Same forward(x, return_all_tokens) interface as OlmoEarthBackbone
    so RenewableEnergyDetector can use either interchangeably.

    The real OlmoEarth model expects MaskedOlmoEarthSample with Sentinel-2 L2A
    data in [B, H, W, T, 12] format (12 spectral bands). This wrapper converts
    our (B, C, H, W) input to that format, keeping everything on GPU.
    """

    # Output dim per model size (from OlmoEarth architecture)
    DIM_MAP = {"nano": 128, "tiny": 384, "base": 768, "large": 1024}
    MODEL_ID_MAP = {
        "nano": "OLMOEARTH_V1_NANO",
        "tiny": "OLMOEARTH_V1_TINY",
        "base": "OLMOEARTH_V1_BASE",
        "300m": "OLMOEARTH_V1_BASE",
        "large": "OLMOEARTH_V1_LARGE",
        "1.4b": "OLMOEARTH_V1_LARGE",
    }
    # OlmoEarth S2 band order: B02,B03,B04,B08, B05,B06,B07,B8A,B11,B12, B01,B09
    # Our pipeline band order: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL,CLD
    # Mapping: our[0,1,2,6,3,4,5,7,8,9] → OlmoEarth[0..9], pad zeros for B01,B09
    _OUR_TO_OLMO_INDICES = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        model_id_name = self.MODEL_ID_MAP.get(config.model_size, "OLMOEARTH_V1_BASE")
        model_id = getattr(ModelID, model_id_name)
        logger.info(f"Loading OlmoEarth pretrained: {model_id.value}")
        self.full_model = load_model_from_id(model_id)
        self.full_model.eval()

        # Get actual output dim from the loaded model
        real_dim = self.DIM_MAP.get(config.model_size, 768)

        # Number of S2 band sets (for mask)
        self._s2_num_band_sets = OlmoModality.SENTINEL2_L2A.num_band_sets  # 3
        self._s2_num_bands = len(OlmoModality.SENTINEL2_L2A.band_order)    # 12

        # Project if real dim != config.hidden_dim
        if real_dim != config.hidden_dim:
            self.projection = nn.Linear(real_dim, config.hidden_dim)
        else:
            self.projection = nn.Identity()

        if config.freeze_backbone:
            for p in self.full_model.parameters():
                p.requires_grad = False

    def _prepare_sample(self, x: torch.Tensor) -> MaskedOlmoEarthSample:
        """Convert (B, C, H, W) → MaskedOlmoEarthSample entirely on GPU."""
        B, C, H, W = x.shape
        device = x.device

        # Resize to 64×64 (OlmoEarth native resolution)
        if H != 64 or W != 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            H, W = 64, 64

        # Reorder our 12 channels to OlmoEarth's 12-band S2 order
        # Our: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12,SCL,CLD
        # OlmoEarth: B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09
        if C >= 10:
            # Take first 10 spectral bands, reorder, pad B01/B09 with zeros
            reordered = x[:, self._OUR_TO_OLMO_INDICES]  # (B, 10, H, W)
            pad = torch.zeros(B, 2, H, W, device=device, dtype=x.dtype)
            x_s2 = torch.cat([reordered, pad], dim=1)  # (B, 12, H, W)
        else:
            # Fewer channels — pad to 12
            pad = torch.zeros(B, self._s2_num_bands - C, H, W, device=device, dtype=x.dtype)
            x_s2 = torch.cat([x, pad], dim=1)

        # BCHW → BHWTC: (B, 12, 64, 64) → (B, 64, 64, 1, 12)
        x_bhwc = x_s2.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_bhwtc = x_bhwc.unsqueeze(3)       # (B, H, W, 1, C)

        # Mask: all tokens visible to encoder → 0 (ONLINE_ENCODER)
        mask = torch.full(
            (B, H, W, 1, self._s2_num_band_sets),
            MaskValue.ONLINE_ENCODER.value,
            device=device, dtype=x.dtype,
        )
        # Mark B01/B09 band set (set 2) as MISSING since we don't have those bands
        mask[:, :, :, :, 2] = MaskValue.MISSING.value

        # Timestamps as int32 (required for month embedding lookup)
        timestamps = torch.tensor(
            [[[15, 6, 2024]]], dtype=torch.int32, device=device
        ).expand(B, -1, -1)

        return MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=x_bhwtc,
            sentinel2_l2a_mask=mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sample = self._prepare_sample(x)
        ctx = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with ctx:
            output = self.full_model.encoder(sample, patch_size=8)

        # tokens shape: (B, H', W', T, S, D) where S = num_band_sets
        features = output["tokens_and_masks"].sentinel2_l2a
        B = features.shape[0]
        D = features.shape[-1]

        # Pool time + band-set dims → (B, H', W', D)
        spatial = features.mean(dim=[3, 4])
        cls_token = self.projection(spatial.mean(dim=[1, 2]))  # (B, D)

        if return_all_tokens:
            patch_tokens = self.projection(spatial.reshape(B, -1, D))  # (B, N, D)
            return cls_token, patch_tokens
        return cls_token


class InstallationDetectionHead(nn.Module):
    """Binary classification: installation exists or not."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class AssetClassificationHead(nn.Module):
    """Multi-class: solar_pv, solar_thermal, wind_onshore, wind_offshore, hydro, none."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, len(EnergySourceType))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CapacityEstimationHead(nn.Module):
    """Regression: estimate capacity in MW from imagery."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class SegmentationHead(nn.Module):
    """Pixel-level segmentation for precise area estimation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.num_patches_side = config.input_size // config.patch_size
        
        # Upsample and project
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, config.num_classes, kernel_size=1)
        )
        
    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) patch embeddings (excluding CLS)
        Returns:
            (B, C, H, W) segmentation logits
        """
        B, N, D = patch_tokens.shape

        # Infer spatial grid size from token count
        H = W = int(N ** 0.5)
        x = patch_tokens.transpose(1, 2).view(B, D, H, W)

        # Decode
        return self.decoder(x)


class RenewableEnergyDetector(nn.Module):
    """
    Complete model for renewable energy detection and verification.
    
    Combines OlmoEarth backbone with task-specific heads for:
    - Installation detection
    - Asset type classification
    - Capacity estimation
    - Segmentation for area calculation
    
    Usage:
        config = ModelConfig()
        model = RenewableEnergyDetector(config, task=TaskType.INSTALLATION_DETECTION)
        
        # Single task
        logits = model(images)
        
        # Multi-task
        model = RenewableEnergyDetector(config, task="multi")
        outputs = model(images)  # Dict with all task outputs
    """
    
    def __init__(
        self, 
        config: ModelConfig,
        task: Union[TaskType, str] = TaskType.INSTALLATION_DETECTION
    ):
        super().__init__()
        self.config = config
        self.task = task

        # Backbone — try real OlmoEarth, fall back to custom ViT
        if config.use_real_backbone and HAS_OLMOEARTH:
            try:
                self.backbone = OlmoEarthRealBackbone(config)
                logger.info(f"Using real OlmoEarth backbone: {config.model_size}")
            except Exception as e:
                logger.warning(f"Real OlmoEarth failed, using custom: {e}")
                self.backbone = OlmoEarthBackbone(config)
        else:
            self.backbone = OlmoEarthBackbone(config)
        
        # Task heads
        if task == "multi" or task == "all":
            self.detection_head = InstallationDetectionHead(config)
            self.classification_head = AssetClassificationHead(config)
            self.capacity_head = CapacityEstimationHead(config)
            self.segmentation_head = SegmentationHead(config)
            self.multi_task = True
        else:
            self.multi_task = False
            if task == TaskType.INSTALLATION_DETECTION:
                self.head = InstallationDetectionHead(config)
            elif task == TaskType.ASSET_CLASSIFICATION:
                self.head = AssetClassificationHead(config)
            elif task == TaskType.CAPACITY_ESTIMATION:
                self.head = CapacityEstimationHead(config)
            else:
                self.head = InstallationDetectionHead(config)
                
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) multi-spectral satellite imagery
            return_features: Also return backbone features
        Returns:
            Task output(s) or dict of outputs for multi-task
        """
        # Get backbone features
        if self.multi_task:
            cls_token, patch_tokens = self.backbone(x, return_all_tokens=True)
        else:
            cls_token = self.backbone(x, return_all_tokens=False)
        
        # Task-specific forward
        if self.multi_task:
            outputs = {
                "detection": torch.sigmoid(self.detection_head(cls_token)),
                "classification": self.classification_head(cls_token),
                "capacity_mw": self.capacity_head(cls_token),
                "segmentation": self.segmentation_head(patch_tokens)
            }
            if return_features:
                outputs["features"] = cls_token
            return outputs
        else:
            output = self.head(cls_token)
            if return_features:
                return output, cls_token
            return output
    
    def freeze_backbone(self, num_layers: Optional[int] = None):
        """Freeze backbone for transfer learning."""
        if num_layers is None:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Freeze first N layers
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            for i, block in enumerate(self.backbone.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
                        
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[ModelConfig] = None,
        task: Union[TaskType, str] = TaskType.INSTALLATION_DETECTION
    ) -> "RenewableEnergyDetector":
        """Load from pretrained checkpoint."""
        config = config or ModelConfig()
        model = cls(config, task)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Handle potential key mismatches
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        return model


def create_model(
    task: str = "detection",
    model_size: str = "base",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    use_real_backbone: bool = True,
) -> RenewableEnergyDetector:
    """
    Factory function to create model.

    Args:
        task: "detection", "classification", "capacity", "segmentation", or "multi"
        model_size: "nano", "tiny", "base", "large"
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to local checkpoint file
        use_real_backbone: Try real OlmoEarth encoder (falls back if unavailable)

    Returns:
        Configured RenewableEnergyDetector model
    """
    dim_map = {"nano": 192, "tiny": 384, "base": 768, "300m": 768, "large": 1024, "1.4b": 1024}
    hidden_dim = dim_map.get(model_size, 768)

    config = ModelConfig(
        model_size=model_size,
        hidden_dim=hidden_dim,
        num_heads=max(1, hidden_dim // 64),
        num_layers=12 if hidden_dim <= 768 else 24,
        use_real_backbone=use_real_backbone and pretrained,
    )

    task_map = {
        "detection": TaskType.INSTALLATION_DETECTION,
        "classification": TaskType.ASSET_CLASSIFICATION,
        "capacity": TaskType.CAPACITY_ESTIMATION,
        "change": TaskType.CHANGE_DETECTION,
        "compliance": TaskType.COMPLIANCE_MONITORING,
        "multi": "multi",
    }
    task_type = task_map.get(task, TaskType.INSTALLATION_DETECTION)

    if pretrained and checkpoint_path:
        return RenewableEnergyDetector.from_pretrained(checkpoint_path, config, task_type)

    return RenewableEnergyDetector(config, task_type)


if __name__ == "__main__":
    # Quick test
    print("Testing RenewableEnergyDetector...")
    
    config = ModelConfig(input_channels=12, input_size=224)
    model = RenewableEnergyDetector(config, task="multi")
    
    # Test forward pass
    x = torch.randn(2, 12, 224, 224)
    outputs = model(x)
    
    print(f"Detection shape: {outputs['detection'].shape}")
    print(f"Classification shape: {outputs['classification'].shape}")
    print(f"Capacity shape: {outputs['capacity_mw'].shape}")
    print(f"Segmentation shape: {outputs['segmentation'].shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

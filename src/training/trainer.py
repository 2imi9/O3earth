"""
Training Pipeline for Renewable Energy Detection Models

Combines:
- EIA ground truth data (plant locations, capacity)
- Satellite imagery (Sentinel-2)
- Fine-tuning of OlmoEarth foundation model

Author: Zim (Millennium Fellowship Research)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training sample."""
    sample_id: str
    image_path: str
    plant_exists: bool
    energy_source: str
    capacity_mw: float
    latitude: float
    longitude: float
    state: Optional[str] = None
    plant_id: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_size: str = "300m"
    task: str = "multi"
    pretrained_path: Optional[str] = None
    
    # Data
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    
    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Augmentation
    augment: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5


class RenewableEnergyDataset(Dataset):
    """Dataset for renewable energy detection."""
    
    def __init__(
        self,
        manifest_path: str,
        transform=None,
        target_size: int = 224
    ):
        self.transform = transform
        self.target_size = target_size
        
        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        self.samples = [TrainingSample(**s) for s in manifest["samples"]]
        
        # Class mappings
        self.energy_sources = ["NONE", "SUN", "WND", "WAT", "GEO"]
        self.source_to_idx = {s: i for i, s in enumerate(self.energy_sources)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = np.load(sample.image_path)  # (C, H, W)
        
        # Resize if needed
        if image.shape[1] != self.target_size:
            image = self._resize(image, self.target_size)
        
        # Normalize
        image = self._normalize(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Labels
        detection_label = torch.tensor(1.0 if sample.plant_exists else 0.0)
        classification_label = torch.tensor(
            self.source_to_idx.get(sample.energy_source, 0)
        )
        capacity_label = torch.tensor(sample.capacity_mw)
        
        return {
            "image": image,
            "detection_label": detection_label,
            "classification_label": classification_label,
            "capacity_label": capacity_label,
            "sample_id": sample.sample_id
        }
    
    def _resize(self, image: np.ndarray, size: int) -> np.ndarray:
        """Simple resize using interpolation."""
        from scipy.ndimage import zoom
        
        c, h, w = image.shape
        zoom_factors = (1, size / h, size / w)
        return zoom(image, zoom_factors, order=1)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range per band."""
        image = image.astype(np.float32)
        for i in range(image.shape[0]):
            p2, p98 = np.percentile(image[i], [2, 98])
            image[i] = np.clip((image[i] - p2) / (p98 - p2 + 1e-8), 0, 1)
        return image


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        classification_weight: float = 1.0,
        capacity_weight: float = 0.1,
        segmentation_weight: float = 1.0
    ):
        super().__init__()
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        self.capacity_weight = capacity_weight
        self.segmentation_weight = segmentation_weight
        
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss.
        
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        
        # Detection loss
        if "detection" in outputs and "detection_label" in targets:
            losses["detection"] = self.bce(
                outputs["detection"],
                targets["detection_label"]
            ) * self.detection_weight
        
        # Classification loss
        if "classification" in outputs and "classification_label" in targets:
            losses["classification"] = self.ce(
                outputs["classification"],
                targets["classification_label"]
            ) * self.classification_weight
        
        # Capacity loss (only for positive samples)
        if "capacity_mw" in outputs and "capacity_label" in targets:
            mask = targets["detection_label"] > 0.5
            if mask.any():
                losses["capacity"] = self.mse(
                    outputs["capacity_mw"][mask],
                    targets["capacity_label"][mask]
                ) * self.capacity_weight
        
        total_loss = sum(losses.values())
        loss_dict = {k: v.item() for k, v in losses.items()}
        
        return total_loss, loss_dict


class Trainer:
    """Training loop for renewable energy detection."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss
        self.criterion = MultiTaskLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # Metrics tracking
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device)
            targets = {
                k: v.to(self.device) 
                for k, v in batch.items() 
                if k != "image" and k != "sample_id"
            }
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            
            pbar.set_postfix(loss=loss.item())
        
        n_batches = len(dataloader)
        return {
            "loss": total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()}
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k != "image" and k != "sample_id"
            }
            
            outputs = self.model(images)
            loss, _ = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Collect predictions
            if "detection" in outputs:
                all_preds.extend((outputs["detection"] > 0.5).cpu().numpy())
                all_labels.extend(targets["detection_label"].cpu().numpy())
        
        # Calculate metrics
        metrics = {"loss": total_loss / len(dataloader)}
        
        if all_preds:
            preds = np.array(all_preds)
            labels = np.array(all_labels)
            metrics["accuracy"] = (preds == labels).mean()
            metrics["precision"] = (preds[labels == 1] == 1).mean() if labels.sum() > 0 else 0
            metrics["recall"] = (labels[preds == 1] == 1).mean() if preds.sum() > 0 else 0
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List]:
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["metrics"].append(val_metrics)
            
            # Step scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            if "accuracy" in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_metrics["loss"])
            
            # Save best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, val_metrics["loss"], is_best=True)
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        
        filename = "best.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")


def create_dataloaders(
    manifest_path: str,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    dataset = RenewableEnergyDataset(manifest_path)
    
    # Split
    n_train = int(len(dataset) * config.train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Training pipeline ready.")
    print("Usage:")
    print("  from src.training.trainer import Trainer, TrainingConfig")
    print("  from src.models.olmo_earth import create_model")
    print("  ")
    print("  model = create_model(task='multi')")
    print("  config = TrainingConfig(epochs=50)")
    print("  trainer = Trainer(model, config)")
    print("  trainer.train(train_loader, val_loader)")

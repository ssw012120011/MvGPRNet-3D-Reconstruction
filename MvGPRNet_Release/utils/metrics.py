"""Evaluation metrics and model verification."""
import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """MSE for 3D tensors (D, H, W)."""
    depth, height, width = predictions.size()
    square_errors = torch.square(predictions - targets)
    mse = torch.sum(square_errors) / (depth * height * width)
    return mse.item()


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return torch.mean(torch.abs(predictions - targets)).item()


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Dice coefficient after binarization."""
    pred_binary = (predictions > threshold).float()
    target_binary = (targets > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    denom = pred_binary.sum() + target_binary.sum()
    return (2 * intersection / denom).item() if denom != 0 else 1.0


def verify_model_n_views(model_path: str, expected_n_views: int) -> bool:
    """Verify model input channels match expected n_views."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if "encoder.conv1.0.weight" in state_dict:
            weight_shape = state_dict["encoder.conv1.0.weight"].shape
            actual_n_views = weight_shape[1]
            return actual_n_views == expected_n_views
        logger.warning("encoder.conv1.0.weight not found in %s", model_path)
        return False
    except Exception as e:
        logger.warning("Error verifying %s: %s", model_path, e)
        return False

"""Multi-view GPR datasets for training and inference."""
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MvGPRDataset(Dataset):
    """Training dataset."""

    def __init__(
        self,
        label_folder: str,
        projection_folder: str,
        n_views: int = 20,
    ) -> None:
        self.label_folder = label_folder
        self.projection_folder = projection_folder
        self.n_views = n_views
        self.label_files = sorted(
            [f for f in os.listdir(label_folder) if f.endswith(".mat")]
        )
        self._build_index()

    def _build_index(self) -> None:
        self.sample_pairs: list[Tuple[str, str]] = []
        for label_f in self.label_files:
            base = label_f.replace(".mat", "")
            proj_f = f"{base}.pkl"
            self.sample_pairs.append((label_f, proj_f))

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label_f, proj_f = self.sample_pairs[idx]
        label_path = os.path.join(self.label_folder, label_f)
        projection_path = os.path.join(self.projection_folder, proj_f)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not os.path.exists(projection_path):
            raise FileNotFoundError(f"Projection file not found: {projection_path}")

        try:
            label_data = loadmat(label_path)["label_data"]
        except Exception as e:
            raise IOError(f"Failed to load .mat file {label_path}: {e}") from e

        label_data = np.nan_to_num(label_data, nan=0.0, posinf=1e5, neginf=-1e5)

        try:
            with open(projection_path, "rb") as f:
                imaging_projs = pickle.load(f)
        except Exception as e:
            raise IOError(
                f"Failed to load projection {projection_path}: {e}"
            ) from e

        imaging_projs = torch.from_numpy(
            np.asarray(imaging_projs, dtype=np.float32)
        )
        label_projs = torch.from_numpy(
            np.asarray(label_data, dtype=np.float32)
        ).unsqueeze(0)

        if torch.isnan(imaging_projs).any() or torch.isinf(imaging_projs).any():
            logger.warning(
                "Sample %d imaging_projs contains NaN/inf, replacing with 0",
                idx,
            )
            imaging_projs = torch.nan_to_num(
                imaging_projs, nan=0.0, posinf=1e5, neginf=-1e5
            )
        if torch.isnan(label_projs).any() or torch.isinf(label_projs).any():
            logger.warning(
                "Sample %d label_projs contains NaN/inf, replacing with 0",
                idx,
            )
            label_projs = torch.nan_to_num(
                label_projs, nan=0.0, posinf=1e5, neginf=-1e5
            )

        return imaging_projs, label_projs


class MvGPRDatasetTest(Dataset):
    """Test dataset (projection files: {base}_noise4000.pkl)."""

    def __init__(
        self,
        label_folder: str,
        projection_folder: str,
        n_views: int = 20,
    ) -> None:
        self.label_folder = label_folder
        self.projection_folder = projection_folder
        self.n_views = n_views
        self.label_files = sorted(
            [f for f in os.listdir(label_folder) if f.endswith(".mat")]
        )
        self._build_index()

    def _build_index(self) -> None:
        self.sample_pairs: list[Tuple[str, str, str]] = []
        for label_f in self.label_files:
            base = label_f.replace(".mat", "")
            proj_f = f"{base}_noise4000.pkl"
            proj_path = os.path.join(self.projection_folder, proj_f)
            if os.path.exists(proj_path):
                self.sample_pairs.append((label_f, proj_f, base))
            else:
                logger.warning(
                    "No projection for label %s (expected %s), skipping",
                    label_f,
                    proj_path,
                )

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        label_f, proj_f, sample_name = self.sample_pairs[idx]
        label_path = os.path.join(self.label_folder, label_f)
        projection_path = os.path.join(self.projection_folder, proj_f)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not os.path.exists(projection_path):
            raise FileNotFoundError(f"Projection file not found: {projection_path}")

        try:
            label_data = loadmat(label_path)["label_data"]
        except Exception as e:
            raise IOError(f"Failed to load .mat file {label_path}: {e}") from e

        label_data = np.nan_to_num(label_data, nan=0.0, posinf=1e5, neginf=-1e5)

        try:
            with open(projection_path, "rb") as f:
                imaging_projs = pickle.load(f)
        except Exception as e:
            raise IOError(
                f"Failed to load projection {projection_path}: {e}"
            ) from e

        imaging_projs = torch.from_numpy(
            np.asarray(imaging_projs, dtype=np.float32)
        )
        label_projs = torch.from_numpy(
            np.asarray(label_data, dtype=np.float32)
        ).unsqueeze(0)

        if torch.isnan(imaging_projs).any() or torch.isinf(imaging_projs).any():
            logger.warning(
                "Sample %d imaging_projs contains NaN/inf, replacing with 0",
                idx,
            )
            imaging_projs = torch.nan_to_num(
                imaging_projs, nan=0.0, posinf=1e5, neginf=-1e5
            )
        if torch.isnan(label_projs).any() or torch.isinf(label_projs).any():
            logger.warning(
                "Sample %d label_projs contains NaN/inf, replacing with 0",
                idx,
            )
            label_projs = torch.nan_to_num(
                label_projs, nan=0.0, posinf=1e5, neginf=-1e5
            )

        return imaging_projs, label_projs, sample_name

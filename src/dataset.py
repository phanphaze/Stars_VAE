# File for loading data into the training loop

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import batch_size, processed_data_path, train_test_split


class ProcessedDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        condition_cols: Optional[Sequence[int]] = None,
    ) -> None:
        self.data = data.astype(np.float32)
        self.condition_cols = tuple(condition_cols or [])
        self.feature_cols = tuple(
            i for i in range(self.data.shape[1]) if i not in self.condition_cols
        )

        self.inputs = torch.from_numpy(self.data[:, self.feature_cols])
        if len(self.condition_cols) > 0:
            self.conditions = torch.from_numpy(self.data[:, self.condition_cols])
        else:
            self.conditions = torch.zeros(
                (len(self.data), 0), dtype=torch.float32
            )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.conditions[idx]


def load_processed_data(path: Path | str = processed_data_path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    return np.load(path)


def create_dataloaders(
    data: Optional[np.ndarray] = None,
    processed_data_path: Path | str = processed_data_path,
    condition_cols: Optional[Sequence[int]] = None,
    batch_size_: int = batch_size,
    train_split: float = train_test_split,
    val_split: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if data is None:
        data = load_processed_data(processed_data_path)

    dataset = ProcessedDataset(data, condition_cols=condition_cols)

    if train_split <= 0 or train_split >= 1:
        raise ValueError("train_split must be between 0 and 1")
    if val_split < 0 or train_split + val_split >= 1:
        raise ValueError("val_split must be >= 0 and train_split + val_split < 1")

    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    splits = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
)
    train_dataset, val_dataset, test_dataset = splits

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_, shuffle=shuffle, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size_, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_, shuffle=False)

    return train_loader, val_loader, test_loader
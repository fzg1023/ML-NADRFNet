"""Dataset utilities for paired 1-D sensor signals stored as NPZ files."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedSignalDataset(Dataset):
    """Load ``x1`` and ``x2`` arrays from class-organized NPZ files.

    Expected layout::

        split_root/
          class_0/*.npz
          class_1/*.npz

    Each NPZ file must contain two one-dimensional arrays named ``x1`` and
    ``x2``. Class indices follow lexicographically sorted directory names.
    """

    def __init__(
        self,
        split_root: str | Path,
        class_names: Sequence[str] | None = None,
        expected_length: int | None = 4096,
        cache: bool = True,
    ) -> None:
        self.root = Path(split_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {self.root}")

        discovered = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_names = list(class_names) if class_names is not None else discovered
        if discovered != self.class_names:
            raise RuntimeError(
                f"Class mismatch in {self.root}. Expected {self.class_names}, found {discovered}"
            )

        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(self.class_names):
            files = sorted((self.root / class_name).glob("*.npz"))
            self.samples.extend((path, label) for path in files)
        if not self.samples:
            raise RuntimeError(f"No NPZ samples found in {self.root}")

        self.expected_length = expected_length
        self.cache = cache
        self.sensor1: torch.Tensor | None = None
        self.sensor2: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
        if cache:
            first, second, labels = [], [], []
            for path, label in self.samples:
                x1, x2 = self._read(path)
                first.append(x1)
                second.append(x2)
                labels.append(label)
            self.sensor1 = torch.from_numpy(np.stack(first)).unsqueeze(1).contiguous()
            self.sensor2 = torch.from_numpy(np.stack(second)).unsqueeze(1).contiguous()
            self.labels = torch.tensor(labels, dtype=torch.long)

    def _read(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as archive:
            if "x1" not in archive or "x2" not in archive:
                raise KeyError(f"{path} must contain arrays named 'x1' and 'x2'")
            x1 = np.asarray(archive["x1"], dtype=np.float32).reshape(-1)
            x2 = np.asarray(archive["x2"], dtype=np.float32).reshape(-1)
        if x1.shape != x2.shape:
            raise ValueError(f"Paired signals have different shapes in {path}")
        if self.expected_length is not None and x1.size != self.expected_length:
            raise ValueError(
                f"Expected length {self.expected_length}, found {x1.size} in {path}"
            )
        return x1, x2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache:
            assert self.sensor1 is not None and self.sensor2 is not None and self.labels is not None
            return self.sensor1[index], self.sensor2[index], self.labels[index]
        path, label = self.samples[index]
        x1, x2 = self._read(path)
        return (
            torch.from_numpy(x1).unsqueeze(0),
            torch.from_numpy(x2).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )

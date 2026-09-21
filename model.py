"""Main dual-sensor 1-D fault-diagnosis network.

The implementation contains only the final method used in the manuscript:
sample-adaptive multi-scale encoding, three-level NA-ADRF fusion, multi-level
feature aggregation, and a compact classifier.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_WIDTHS: Tuple[int, int, int, int] = (28, 56, 112, 232)


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class SeparableScaleBranch1D(nn.Module):
    """Depthwise-separable temporal branch for one receptive-field scale."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.layers(signal)


class SampleAdaptiveMultiScaleBlock1D(nn.Module):
    """Aggregate 7-, 15-, and 31-tap branches with sample-dependent weights."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            SeparableScaleBranch1D(in_channels, out_channels, kernel_size)
            for kernel_size in (7, 15, 31)
        )
        hidden_channels = max(8, in_channels // 4)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, 3, 1),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        branch_features = torch.stack(
            [branch(signal) for branch in self.branches], dim=1
        )
        weights = F.softmax(self.weight_generator(signal).squeeze(-1), dim=1)
        weights = weights.view(signal.shape[0], 3, 1, 1)
        fused = (branch_features * weights).sum(dim=1)
        return F.relu(fused + self.residual(signal), inplace=True)


class SensorEncoder1D(nn.Module):
    """Sample-adaptive encoder that returns shallow, middle, and deep features."""

    def __init__(self, widths: Tuple[int, int, int, int] = DEFAULT_WIDTHS) -> None:
        super().__init__()
        stem, block1, block2, deep = widths
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem, 3, padding=1, bias=False),
            nn.BatchNorm1d(stem),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool1d(2)
        self.block1 = SampleAdaptiveMultiScaleBlock1D(stem, block1)
        self.block2 = SampleAdaptiveMultiScaleBlock1D(block1, block2)
        self.block3 = SampleAdaptiveMultiScaleBlock1D(block2, deep)
        self.block4 = SampleAdaptiveMultiScaleBlock1D(deep, deep)

    def forward(self, signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        shallow = self.pool(self.stem(signal))
        x = self.pool(self.block1(shallow))
        middle = self.block2(x)
        x = self.pool(middle)
        x = self.pool(self.block3(x))
        deep = self.block4(x)
        return {"shallow": shallow, "middle": middle, "deep": deep}


class SensorResidualAdapter1D(nn.Module):
    """Sensor-specific bottleneck adapter with an identity-preserving path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = min(64, max(8, channels // 2))
        self.transform = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1, bias=False),
            nn.GroupNorm(_group_count(bottleneck), bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.transform(features), inplace=True)


class NADRFFusion1D(nn.Module):
    """Normalized Agreement-Discrepancy Residual Fusion (NA-ADRF)."""

    def __init__(self, channels: int, shared_channels: int | None = None) -> None:
        super().__init__()
        if shared_channels is None:
            shared_channels = 128 if channels >= 232 else 64 if channels >= 112 else 16

        self.base_fusion = nn.Sequential(
            nn.Conv1d(2 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.sensor1_adapter = SensorResidualAdapter1D(channels)
        self.sensor2_adapter = SensorResidualAdapter1D(channels)
        self.shared_projector = nn.Sequential(
            nn.Conv1d(channels, shared_channels, 1, bias=False),
            nn.GroupNorm(_group_count(shared_channels), shared_channels),
            nn.ReLU(inplace=True),
        )
        self.relation_transform = nn.Sequential(
            nn.Conv1d(2 * shared_channels, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        # ReZero initialization preserves the stable concatenation path at the
        # beginning of training and learns the relation residual gradually.
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, sensor1_features: torch.Tensor, sensor2_features: torch.Tensor
    ) -> torch.Tensor:
        base = self.base_fusion(
            torch.cat((sensor1_features, sensor2_features), dim=1)
        )
        sensor1_shared = self.shared_projector(
            self.sensor1_adapter(sensor1_features)
        )
        sensor2_shared = self.shared_projector(
            self.sensor2_adapter(sensor2_features)
        )
        sensor1_shared = F.normalize(sensor1_shared, p=2, dim=1, eps=1e-6)
        sensor2_shared = F.normalize(sensor2_shared, p=2, dim=1, eps=1e-6)

        agreement = sensor1_shared * sensor2_shared
        discrepancy = (sensor1_shared - sensor2_shared).square()
        relation = self.relation_transform(
            torch.cat((agreement, discrepancy), dim=1)
        )
        return F.relu(base + self.beta * relation, inplace=True)


class MultiSensorFaultDiagnosisNet(nn.Module):
    """Final dual-stream sample-adaptive multi-level NA-ADRF network."""

    def __init__(
        self,
        num_classes: int,
        widths: Tuple[int, int, int, int] = DEFAULT_WIDTHS,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        stem, _, middle, deep = widths
        self.num_classes = num_classes
        self.sensor1_encoder = SensorEncoder1D(widths)
        self.sensor2_encoder = SensorEncoder1D(widths)
        self.shallow_fusion = NADRFFusion1D(stem)
        self.middle_fusion = NADRFFusion1D(middle)
        self.deep_fusion = NADRFFusion1D(deep)
        self.channel_bottleneck = nn.Sequential(
            nn.Conv1d(stem + middle + deep, deep, 1, bias=False),
            nn.BatchNorm1d(deep),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(deep, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, sensor1_signal: torch.Tensor, sensor2_signal: torch.Tensor
    ) -> torch.Tensor:
        if sensor1_signal.ndim != 3 or sensor2_signal.ndim != 3:
            raise ValueError("Inputs must have shape [batch, 1, signal_length]")
        if sensor1_signal.shape != sensor2_signal.shape:
            raise ValueError("The two sensor inputs must have identical shapes")

        sensor1 = self.sensor1_encoder(sensor1_signal)
        sensor2 = self.sensor2_encoder(sensor2_signal)
        shallow = self.shallow_fusion(sensor1["shallow"], sensor2["shallow"])
        middle = self.middle_fusion(sensor1["middle"], sensor2["middle"])
        deep = self.deep_fusion(sensor1["deep"], sensor2["deep"])

        target_length = deep.shape[-1]
        aggregated = torch.cat(
            (
                F.adaptive_avg_pool1d(shallow, target_length),
                F.adaptive_avg_pool1d(middle, target_length),
                deep,
            ),
            dim=1,
        )
        refined = self.channel_bottleneck(aggregated)
        pooled = self.global_pool(refined).squeeze(-1)
        return self.classification_head(pooled)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    network = MultiSensorFaultDiagnosisNet(num_classes=9)
    sample = torch.randn(2, 1, 4096)
    prediction = network(sample, sample)
    print(f"Output shape: {tuple(prediction.shape)}")
    print(f"Trainable parameters: {network.trainable_parameter_count:,}")

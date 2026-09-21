"""Signal-level noise augmentation and normalization."""
from __future__ import annotations

import torch


def zscore(signal: torch.Tensor) -> torch.Tensor:
    """Apply sample-wise Z-score normalization along the temporal dimension."""
    mean = signal.mean(dim=-1, keepdim=True)
    standard_deviation = signal.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (signal - mean) / standard_deviation


def add_awgn(
    signal: torch.Tensor,
    snr_low_db: float,
    snr_high_db: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add independent AWGN at one uniformly sampled SNR per sample."""
    if snr_low_db > snr_high_db:
        raise ValueError("snr_low_db must not exceed snr_high_db")
    batch_size = signal.shape[0]
    snr_db = torch.empty(
        (batch_size, 1, 1), device=signal.device, dtype=signal.dtype
    ).uniform_(snr_low_db, snr_high_db)
    noise = torch.randn(
        signal.shape, device=signal.device, dtype=signal.dtype
    )
    signal_rms = signal.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    noise_rms = noise.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    target_noise_rms = signal_rms / torch.pow(signal.new_tensor(10.0), snr_db / 20.0)
    noisy = signal + noise * (target_noise_rms / noise_rms)
    return noisy, snr_db.flatten()


def add_awgn_then_zscore(
    signal: torch.Tensor,
    snr_low_db: float,
    snr_high_db: float,
) -> torch.Tensor:
    noisy, _ = add_awgn(signal, snr_low_db, snr_high_db)
    return zscore(noisy)


def apply_modality_dropout(
    sensor1: torch.Tensor,
    sensor2: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop at most one sensor per selected sample."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Modality-dropout probability must be in [0, 1]")
    if probability == 0.0:
        return sensor1, sensor2
    draw = torch.rand(
        (sensor1.shape[0], 1, 1), device=sensor1.device
    )
    drop_sensor1 = draw < probability / 2.0
    drop_sensor2 = (draw >= probability / 2.0) & (draw < probability)
    return (
        sensor1.masked_fill(drop_sensor1, 0.0),
        sensor2.masked_fill(drop_sensor2, 0.0),
    )

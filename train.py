"""Train the final dual-sensor NA-ADRF model."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from augmentation import add_awgn_then_zscore, apply_modality_dropout
from data import PairedSignalDataset
from metrics import metrics_from_confusion_matrix, update_confusion_matrix
from model import MultiSensorFaultDiagnosisNet
from utils import resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train/val/test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/main_method"))
    parser.add_argument("--signal-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--snr-low", type=float, default=-15.0)
    parser.add_argument("--snr-high", type=float, default=-4.0)
    parser.add_argument("--modality-dropout", type=float, default=0.20)
    parser.add_argument("--disable-online-noise", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true", help="Load NPZ files on demand")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    return parser.parse_args()


def make_loader(
    dataset: PairedSignalDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for sensor1, sensor2, target in loader:
        logits = model(
            sensor1.to(device, non_blocking=True),
            sensor2.to(device, non_blocking=True),
        )
        update_confusion_matrix(confusion, target, logits.argmax(dim=1))
    return metrics_from_confusion_matrix(confusion)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = PairedSignalDataset(
        args.data_root / "train",
        expected_length=args.signal_length,
        cache=not args.no_cache,
    )
    validation_set = PairedSignalDataset(
        args.data_root / "val",
        class_names=train_set.class_names,
        expected_length=args.signal_length,
        cache=not args.no_cache,
    )
    test_set = PairedSignalDataset(
        args.data_root / "test",
        class_names=train_set.class_names,
        expected_length=args.signal_length,
        cache=not args.no_cache,
    )
    validation_loader = make_loader(
        validation_set, args.batch_size, args.workers, False
    )
    test_loader = make_loader(test_set, args.batch_size, args.workers, False)

    model = MultiSensorFaultDiagnosisNet(len(train_set.class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    loss_function = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)

    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    print(
        f"Device={device} | classes={len(train_set.class_names)} | "
        f"parameters={model.trainable_parameter_count:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loader = make_loader(
            train_set, args.batch_size, args.workers, True
        )
        running_loss = 0.0
        correct = 0
        seen = 0
        for sensor1, sensor2, target in train_loader:
            sensor1 = sensor1.to(device, non_blocking=True)
            sensor2 = sensor2.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if not args.disable_online_noise:
                sensor1 = add_awgn_then_zscore(sensor1, args.snr_low, args.snr_high)
                sensor2 = add_awgn_then_zscore(sensor2, args.snr_low, args.snr_high)
            sensor1, sensor2 = apply_modality_dropout(
                sensor1,
                sensor2,
                args.modality_dropout,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(sensor1, sensor2)
                loss = loss_function(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = target.numel()
            running_loss += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == target).sum())
            seen += batch_size

        train_loss = running_loss / max(seen, 1)
        train_accuracy = correct / max(seen, 1)
        validation_metrics = evaluate(
            model, validation_loader, device, len(train_set.class_names)
        )
        validation_accuracy = validation_metrics["accuracy"]
        scheduler.step(validation_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
        )
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"train={100 * train_accuracy:.2f}% | val={100 * validation_accuracy:.2f}%"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, len(train_set.class_names))

    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": train_set.class_names,
        "signal_length": args.signal_length,
        "num_classes": len(train_set.class_names),
        "trainable_parameters": model.trainable_parameter_count,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_metrics": test_metrics,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    write_json(
        args.output_dir / "summary.json",
        {
            "classes": train_set.class_names,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "test_metrics": test_metrics,
            "trainable_parameters": model.trainable_parameter_count,
        },
    )
    print(f"Best epoch: {best_epoch}")
    print("Test metrics:", test_metrics)
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

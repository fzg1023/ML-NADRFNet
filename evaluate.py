"""Evaluate a trained model on a paired-signal test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from augmentation import add_awgn_then_zscore, zscore
from data import PairedSignalDataset
from metrics import metrics_from_confusion_matrix, update_confusion_matrix
from model import MultiSensorFaultDiagnosisNet
from utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sensor1-snr", type=float, default=None)
    parser.add_argument("--sensor2-snr", type=float, default=None)
    parser.add_argument("--missing-sensor", choices=("none", "1", "2"), default="none")
    parser.add_argument("--zscore-input", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    signal_length = int(checkpoint.get("signal_length", 4096))
    dataset = PairedSignalDataset(
        args.test_root,
        class_names=class_names,
        expected_length=signal_length,
        cache=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = MultiSensorFaultDiagnosisNet(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)

    with torch.inference_mode():
        for sensor1, sensor2, target in loader:
            sensor1 = sensor1.to(device, non_blocking=True)
            sensor2 = sensor2.to(device, non_blocking=True)
            if args.sensor1_snr is not None:
                sensor1 = add_awgn_then_zscore(
                    sensor1, args.sensor1_snr, args.sensor1_snr
                )
            elif args.zscore_input:
                sensor1 = zscore(sensor1)
            if args.sensor2_snr is not None:
                sensor2 = add_awgn_then_zscore(
                    sensor2, args.sensor2_snr, args.sensor2_snr
                )
            elif args.zscore_input:
                sensor2 = zscore(sensor2)
            if args.missing_sensor == "1":
                sensor1 = torch.zeros_like(sensor1)
            elif args.missing_sensor == "2":
                sensor2 = torch.zeros_like(sensor2)

            prediction = model(sensor1, sensor2).argmax(dim=1)
            update_confusion_matrix(confusion, target, prediction)

    results = metrics_from_confusion_matrix(confusion)
    results["classes"] = class_names
    results["confusion_matrix"] = confusion.tolist()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

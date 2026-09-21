# Dual-Sensor Fault Diagnosis with Sample-Adaptive Multi-Scale Encoding and NA-ADRF

This repository contains the clean implementation of the main method described in our manuscript. It accepts two synchronized one-dimensional sensor signals and performs end-to-end fault classification using:

1. dual-stream sample-adaptive multi-scale encoders;
2. Normalized Agreement-Discrepancy Residual Fusion (NA-ADRF) at shallow, middle, and deep feature levels;
3. multi-level feature alignment and aggregation; and
4. a compact classification head.

Only the final method is included. Datasets, trained weights, generated results, figures, logs, local paths, and ablation-only networks are intentionally excluded.

## Repository structure

```text
paper1_code/
├── model.py          # Final network and NA-ADRF implementation
├── data.py           # Paired-signal NPZ dataset loader
├── augmentation.py   # Online AWGN, Z-score normalization, modality dropout
├── metrics.py        # Accuracy, macro precision/recall/F1, balanced accuracy
├── utils.py          # Reproducibility and device utilities
├── train.py          # Training, validation, early stopping, checkpoint export
├── evaluate.py       # Testing, asymmetric-noise and missing-sensor evaluation
├── requirements.txt
└── README.md
```

## Method overview

Each sensor stream is processed by an independent encoder. Every encoding block contains depthwise-separable convolution branches with kernel sizes 7, 15, and 31. A sample-dependent Softmax weighting network determines the contribution of each receptive-field scale.

At the shallow, middle, and deep levels, NA-ADRF first preserves a conventional concatenation-based residual path. Sensor-specific adapters then reduce sensor-domain differences, after which a weight-shared projector maps both streams into a common feature space. Channel-wise L2 normalization is applied before constructing agreement and discrepancy cues:

```text
agreement   = normalized_sensor1 * normalized_sensor2
discrepancy = (normalized_sensor1 - normalized_sensor2)^2
```

The transformed relation feature is introduced through a learnable ReZero scalar `beta`. The three fused features are aligned to the deep temporal resolution, concatenated, channel-compressed, globally pooled, and classified.

With nine output classes, the default network contains **1,412,428 trainable parameters**. The exact count changes slightly with the number of classes.

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- NumPy 1.24 or later
- CUDA-capable GPU recommended but not required

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Data format

The training command expects the following directory layout:

```text
data_root/
├── train/
│   ├── class_01/*.npz
│   ├── class_02/*.npz
│   └── ...
├── val/
│   ├── class_01/*.npz
│   └── ...
└── test/
    ├── class_01/*.npz
    └── ...
```

Every NPZ file must contain two synchronized, one-dimensional `float32` arrays:

```python
import numpy as np

np.savez_compressed(
    "segment_001.npz",
    x1=sensor_1_signal.astype(np.float32),
    x2=sensor_2_signal.astype(np.float32),
)
```

The default signal length is 4096 samples. Class labels are assigned according to lexicographically sorted class-directory names, and the same class directories must be present in all three splits.

### Leakage-safe splitting

Split each continuous recording into non-overlapping temporal blocks for training, validation, and testing **before** extracting windows. Sliding windows may overlap within a split, but windows from different splits must not share raw samples. A guard interval of at least one window length between adjacent splits is recommended. If multiple independent recordings are available, recording-level splitting is preferred.

The paper protocol uses 70 training, 30 validation, and 30 testing windows per class, a window length of 4096, and 25% overlap within each split. These counts describe the experimental protocol, not a requirement imposed by the code.

## Training

The default command reproduces the main training configuration:

```bash
python train.py \
  --data-root path/to/data_root \
  --output-dir outputs/run_01 \
  --epochs 60 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --snr-low -15 \
  --snr-high -4 \
  --modality-dropout 0.2
```

During training, each sensor window independently receives white Gaussian noise. One SNR is sampled uniformly from `[-15, -4]` dB for every sample and epoch, after which sample-wise Z-score normalization is applied. Modality dropout selects at most one sensor per affected sample; the default total dropout probability is 0.2.

Validation and test arrays are used exactly as stored. For the paper protocol, they should therefore be generated from the corresponding raw temporal blocks using the intended fixed evaluation noise and then Z-score normalized.

To train without online noise augmentation, add:

```bash
--disable-online-noise
```

## Evaluation

Evaluate a saved model on a prepared test split:

```bash
python evaluate.py \
  --checkpoint outputs/run_01/best_model.pt \
  --test-root path/to/data_root/test \
  --output outputs/run_01/test_metrics.json
```

The script reports accuracy, macro precision, macro recall, macro F1, balanced accuracy, and the confusion matrix.

### Asymmetric sensor noise

For a **clean, unnormalized** test set, fixed SNRs can be applied at evaluation time:

```bash
python evaluate.py \
  --checkpoint outputs/run_01/best_model.pt \
  --test-root path/to/clean_test \
  --sensor1-snr -6 \
  --sensor2-snr -15
```

Do not apply these options to data that already contain artificial noise, because doing so adds noise twice. For exact manuscript reproduction, fixed noisy test sets should be generated once from the original clean temporal blocks and reused by every comparison method.

### Missing-sensor evaluation

Set one complete sensor input to zero:

```bash
python evaluate.py \
  --checkpoint outputs/run_01/best_model.pt \
  --test-root path/to/prepared_test \
  --missing-sensor 2
```

## Outputs

Training writes only the following files to the selected output directory:

- `best_model.pt`: best validation checkpoint, class names, configuration, and test metrics;
- `history.csv`: epoch-level training and validation history;
- `summary.json`: compact validation/test summary and parameter count.

These generated files are ignored by Git and are not included in this repository.

## Reproducibility

Run the training command independently multiple times and report the mean and sample standard deviation across the resulting models. Model selection must use validation performance only; the test set must not be used to select checkpoints or configurations.

Exact results may vary across runs and can also depend on the PyTorch version, CUDA version, and GPU model.

## Public datasets

The repository does not redistribute third-party data. Download the datasets from their official providers and follow their respective licenses and citation requirements. Convert the two synchronized sensor channels into the NPZ layout described above.

## Citation

If this code contributes to your work, please cite the associated article. Replace the placeholder below with the final bibliographic information after publication:

```bibtex
@article{author2026multisensor,
  title   = {Title of the associated article},
  author  = {Author list},
  journal = {Journal name},
  year    = {2026}
}
```

## License

No license is assigned in this cleaned folder because the authors must select a license consistent with the datasets, institutional policy, and target journal. Before public release, add an explicit open-source license (for example, MIT or BSD-3-Clause) and update this section accordingly.

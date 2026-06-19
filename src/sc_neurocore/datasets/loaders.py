# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic dataset loaders with synthetic fallbacks

"""Neuromorphic dataset loaders with synthetic fallbacks.

Supports N-MNIST, Spiking Heidelberg Digits (SHD), and DVS-CIFAR10.
When real data is unavailable, generates reproducible synthetic spike
patterns via Poisson encoding for pipeline testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .encoding import poisson_encode

# N-MNIST resolution and download source
_NMNIST_RES = 34
_NMNIST_URL = "https://www.garrickorchard.com/datasets/n-mnist"

# SHD channel count and download source
_SHD_CHANNELS = 700
_SHD_URL = "https://zenkelab.org/datasets/"

# DVS-CIFAR10 resolution and download source
_DVS_CIFAR10_RES = 128
_DVS_CIFAR10_URL = "https://figshare.com/articles/dataset/CIFAR10-DVS/4724671"


def _synthetic_event_dataset(
    n_samples: int,
    spatial_size: int,
    n_classes: int,
    T: int,
    dt_ms: float,
    seed: int,
) -> tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Generate synthetic Poisson-encoded event samples.

    Each class gets a distinct random rate template. Events are returned
    as (N_events, 4) arrays with columns [x, y, polarity, timestamp_ms].
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_samples)
    templates = rng.uniform(0.0, 0.3, size=(n_classes, spatial_size, spatial_size))

    samples: list[np.ndarray[Any, Any]] = []
    for i in range(n_samples):
        rates = templates[labels[i]].ravel()
        spike_train = poisson_encode(rates, T, dt_ms=dt_ms, seed=seed + i + 1)
        timesteps, neuron_ids = np.nonzero(spike_train)
        y_coords, x_coords = np.divmod(neuron_ids, spatial_size)
        polarities = rng.integers(0, 2, size=len(timesteps))
        events = np.column_stack(
            [
                x_coords,
                y_coords,
                polarities,
                timesteps * dt_ms,
            ]
        ).astype(np.float32)
        samples.append(events)

    return samples, labels


def _check_root(root: str | Path, dataset_name: str, url: str) -> Path:
    """Raise FileNotFoundError if *root* does not exist."""
    p = Path(root)
    if p.exists():
        return p
    raise FileNotFoundError(f"{dataset_name}: download from {url} into {p}")


def load_nmnist(
    root: str | Path = "data/nmnist",
    train: bool = True,
    dt_ms: float = 1.0,
    T: int = 300,
    synthetic: bool = False,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Load N-MNIST spiking vision dataset.

    Neuromorphic-MNIST: 34x34 DVS recordings of MNIST digits moved on
    an ATIS sensor via saccadic eye movements. 10 classes.

    Orchard et al., "Converting Static Image Datasets to Spiking
    Neuromorphic Datasets Using Saccades", Front. Neurosci. 2015.

    Parameters
    ----------
    root : path
        Directory containing the extracted dataset.
    train : bool
        Load training split if True, test split otherwise.
    dt_ms : float
        Temporal resolution for synthetic fallback.
    T : int
        Number of timesteps for synthetic fallback.
    synthetic : bool
        Force synthetic data generation.
    n_samples : int
        Number of synthetic samples to generate.
    seed : int
        RNG seed for reproducible synthetic data.

    Returns
    -------
    samples : list of ndarray, each shape (N_events, 4)
        Columns: [x, y, polarity, timestamp_ms].
    labels : ndarray of int
    """
    if synthetic:
        return _synthetic_event_dataset(
            n_samples,
            _NMNIST_RES,
            10,
            T,
            dt_ms,
            seed,
        )
    _check_root(root, "N-MNIST", _NMNIST_URL)
    split_dir = Path(root) / ("Train" if train else "Test")
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected split directory {split_dir.resolve()}. Download from {_NMNIST_URL}"
        )
    # Real loader: N-MNIST uses .bin files, one per sample, grouped by class
    samples: list[np.ndarray[Any, Any]] = []
    label_list: list[int] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = int(class_dir.name)
        for bin_file in sorted(class_dir.glob("*.bin")):
            events = _parse_nmnist_bin(bin_file, dt_ms)
            samples.append(events)
            label_list.append(class_label)
    return samples, np.array(label_list, dtype=np.int64)


def _parse_nmnist_bin(path: Path, dt_ms: float) -> np.ndarray[Any, Any]:
    """Parse a single N-MNIST .bin file into (N, 4) event array."""
    raw = np.fromfile(path, dtype=np.uint8)
    # Each event is 5 bytes: [addr_high, addr_low, ts2, ts1, ts0]
    n_events = len(raw) // 5
    raw = raw[: n_events * 5].reshape(n_events, 5)
    addr = (raw[:, 0].astype(np.uint16) << 8) | raw[:, 1].astype(np.uint16)
    x = addr & 0x1F  # bits 0-4
    y = (addr >> 5) & 0x1F  # bits 5-9
    polarity = (addr >> 10) & 0x1  # bit 10
    ts = (
        raw[:, 2].astype(np.uint32) << 16
        | raw[:, 3].astype(np.uint32) << 8
        | raw[:, 4].astype(np.uint32)
    )
    ts_ms = ts.astype(np.float32) * (dt_ms / 1000.0)
    return np.column_stack([x, y, polarity, ts_ms]).astype(np.float32)


def load_shd(
    root: str | Path = "data/shd",
    train: bool = True,
    dt_ms: float = 1.0,
    T: int = 1000,
    synthetic: bool = False,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Load Spiking Heidelberg Digits (SHD) dataset.

    Audio digits 0-9 in English and German, spike-encoded through an
    artificial cochlea model. 700 input channels, 20 classes.

    Cramer et al., "The Heidelberg Spiking Data Sets for the Systematic
    Evaluation of Spiking Neural Networks", IEEE TNNLS 2022.

    Parameters
    ----------
    root : path
        Directory containing shd_train.h5 / shd_test.h5.
    train : bool
        Load training split if True, test split otherwise.
    dt_ms : float
        Temporal resolution for binning spikes.
    T : int
        Number of timesteps for synthetic fallback.
    synthetic : bool
        Force synthetic data generation.
    n_samples : int
        Number of synthetic samples to generate.
    seed : int
        RNG seed for reproducible synthetic data.

    Returns
    -------
    samples : list of ndarray, each shape (T, 700) dtype bool
        Binned spike trains.
    labels : ndarray of int
    """
    if synthetic:
        return _synthetic_shd(n_samples, T, dt_ms, seed)

    _check_root(root, "SHD", _SHD_URL)
    fname = "shd_train.h5" if train else "shd_test.h5"
    h5_path = Path(root) / fname
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path.resolve()} not found. Download from {_SHD_URL}")
    import h5py

    samples: list[np.ndarray[Any, Any]] = []
    with h5py.File(h5_path, "r") as f:
        spike_times = f["spikes"]["times"]
        spike_units = f["spikes"]["units"]
        raw_labels = f["labels"][:]
        for i in range(len(raw_labels)):
            times = np.asarray(spike_times[i])
            units = np.asarray(spike_units[i])
            if len(times) > 0:
                n_bins = min(int(np.ceil(times.max() / (dt_ms / 1000.0))) + 1, T)
            else:
                n_bins = T
            train_arr = np.zeros((n_bins, _SHD_CHANNELS), dtype=bool)
            if len(times) > 0:
                bin_idx = np.clip((times / (dt_ms / 1000.0)).astype(int), 0, n_bins - 1)
                unit_idx = np.clip(units.astype(int), 0, _SHD_CHANNELS - 1)
                train_arr[bin_idx, unit_idx] = True
            samples.append(train_arr)

    return samples, raw_labels.astype(np.int64)


def _synthetic_shd(
    n_samples: int,
    T: int,
    dt_ms: float,
    seed: int,
) -> tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 20, size=n_samples)
    templates = rng.uniform(0.0, 0.1, size=(20, _SHD_CHANNELS))
    samples: list[np.ndarray[Any, Any]] = []
    for i in range(n_samples):
        spike_train = poisson_encode(
            templates[labels[i]],
            T,
            dt_ms=dt_ms,
            seed=seed + i + 1,
        )
        samples.append(spike_train)
    return samples, labels


def load_dvs_cifar10(
    root: str | Path = "data/dvs_cifar10",
    train: bool = True,
    dt_ms: float = 1.0,
    T: int = 300,
    synthetic: bool = False,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[list[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Load DVS-CIFAR10 event-camera dataset.

    CIFAR-10 images displayed on a monitor and recorded by a DVS camera
    at 128x128 resolution. 10 classes.

    Li et al., "CIFAR10-DVS: An Event-Stream Dataset for Object
    Classification", Front. Neurosci. 2017.

    Parameters
    ----------
    root : path
        Directory containing the extracted dataset.
    train : bool
        Load training split if True, test split otherwise.
    dt_ms : float
        Temporal resolution for synthetic fallback.
    T : int
        Number of timesteps for synthetic fallback.
    synthetic : bool
        Force synthetic data generation.
    n_samples : int
        Number of synthetic samples to generate.
    seed : int
        RNG seed for reproducible synthetic data.

    Returns
    -------
    samples : list of ndarray, each shape (N_events, 4)
        Columns: [x, y, polarity, timestamp_ms].
    labels : ndarray of int
    """
    if synthetic:
        return _synthetic_event_dataset(
            n_samples,
            _DVS_CIFAR10_RES,
            10,
            T,
            dt_ms,
            seed,
        )
    _check_root(root, "DVS-CIFAR10", _DVS_CIFAR10_URL)
    split_dir = Path(root) / ("train" if train else "test")
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected split directory {split_dir.resolve()}. Download from {_DVS_CIFAR10_URL}"
        )
    # Real loader: .aedat or .mat files grouped by class
    samples: list[np.ndarray[Any, Any]] = []
    label_list: list[int] = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = int(class_dir.name)
        for event_file in sorted(class_dir.glob("*.npy")):
            events = np.load(event_file).astype(np.float32)
            samples.append(events)
            label_list.append(class_label)
    if not samples:
        raise FileNotFoundError(
            f"No .npy event files found in {split_dir.resolve()}. "
            f"Convert raw data to .npy arrays with columns [x, y, pol, ts_ms]."
        )
    return samples, np.array(label_list, dtype=np.int64)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS event camera pipeline

"""DVS (Dynamic Vision Sensor) event processing pipeline.

Load event camera data, convert to spike trains or frames, and feed
into SC-NeuroCore networks for processing and FPGA deployment.

Supports raw event arrays (structured numpy) and integration with
the Tonic library for standard DVS datasets (N-MNIST, DVS-Gesture,
N-Cars, etc.).

Event format: structured array with fields (x, y, t, p)
  x: pixel x coordinate
  y: pixel y coordinate
  t: timestamp (microseconds)
  p: polarity (0=OFF, 1=ON)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DVSLoader:
    """Load and preprocess DVS event camera data.

    Parameters
    ----------
    width : int
        Sensor width in pixels.
    height : int
        Sensor height in pixels.
    """

    width: int = 346
    height: int = 260

    @property
    def n_pixels(self) -> int:
        """Return the total number of pixels in the DVS frame."""
        return self.width * self.height

    def from_numpy(self, events: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Load events from structured numpy array.

        Expected fields: 'x', 'y', 't', 'p' (or positional columns).
        Returns structured array with named fields.
        """
        if events.dtype.names is not None:
            return events
        if events.ndim == 2 and events.shape[1] >= 4:
            dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
            structured = np.zeros(events.shape[0], dtype=dtype)
            structured["x"] = events[:, 0].astype(np.int32)
            structured["y"] = events[:, 1].astype(np.int32)
            structured["t"] = events[:, 2].astype(np.int64)
            structured["p"] = events[:, 3].astype(np.int8)
            return structured
        raise ValueError("Events must be structured array or (N, 4+) array with x, y, t, p columns")

    def from_tonic(self, dataset_name: str, index: int = 0) -> tuple[np.ndarray[Any, Any], int]:
        """Load events from a Tonic dataset (requires tonic package).

        Parameters
        ----------
        dataset_name : str
            Tonic dataset name: 'nmnist', 'dvs_gesture', 'ncars', etc.
        index : int
            Sample index in the dataset.

        Returns
        -------
        (events, target) tuple
        """
        try:
            import tonic
        except ImportError:
            raise ImportError("pip install tonic") from None

        dataset_map = {  # pragma: no cover
            "nmnist": tonic.datasets.NMNIST,
            "dvs_gesture": tonic.datasets.DVSGesture,
        }
        cls = dataset_map.get(dataset_name)  # pragma: no cover
        if cls is None:  # pragma: no cover
            raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(dataset_map)}")

        ds = cls(save_to="./data", train=True)  # pragma: no cover
        events, target = ds[index]  # pragma: no cover
        return self.from_numpy(events), target  # pragma: no cover


def events_to_spike_trains(
    events: np.ndarray[Any, Any],
    width: int,
    height: int,
    dt_us: float = 1000.0,
    duration_us: float | None = None,
) -> np.ndarray[Any, Any]:
    """Convert DVS events to binary spike train matrix.

    Parameters
    ----------
    events : structured ndarray with x, y, t, p fields
    width, height : int
        Sensor dimensions.
    dt_us : float
        Time bin width in microseconds (default 1000 = 1ms).
    duration_us : float, optional
        Total duration. If None, inferred from event timestamps.

    Returns
    -------
    ndarray of shape (n_bins, width * height * 2)
        Binary spike trains. Channels: [ON pixels, OFF pixels].
    """
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    p = events["p"].astype(np.int8)

    t_min = t.min()
    t_rel = t - t_min

    if duration_us is None:
        duration_us = t_rel.max() + dt_us

    n_bins = max(1, int(np.ceil(duration_us / dt_us)))
    n_channels = width * height * 2

    spikes = np.zeros((n_bins, n_channels), dtype=np.int8)

    for i in range(len(events)):
        bin_idx = min(int(t_rel[i] / dt_us), n_bins - 1)
        pixel_idx = int(y[i]) * width + int(x[i])
        if p[i] > 0:
            channel = pixel_idx
        else:
            channel = width * height + pixel_idx
        if 0 <= channel < n_channels:
            spikes[bin_idx, channel] = 1

    return spikes


def events_to_frames(
    events: np.ndarray[Any, Any],
    width: int,
    height: int,
    dt_us: float = 10000.0,
    duration_us: float | None = None,
) -> np.ndarray[Any, Any]:
    """Convert DVS events to event count frames.

    Parameters
    ----------
    events : structured ndarray
    width, height : int
    dt_us : float
        Frame duration in microseconds (default 10000 = 10ms).
    duration_us : float, optional

    Returns
    -------
    ndarray of shape (n_frames, 2, height, width)
        Event count frames with ON and OFF channels.
    """
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    p = events["p"].astype(np.int8)

    t_min = t.min()
    t_rel = t - t_min

    if duration_us is None:
        duration_us = t_rel.max() + dt_us

    n_frames = max(1, int(np.ceil(duration_us / dt_us)))
    frames = np.zeros((n_frames, 2, height, width), dtype=np.float32)

    for i in range(len(events)):
        f = min(int(t_rel[i] / dt_us), n_frames - 1)
        yi = min(int(y[i]), height - 1)
        xi = min(int(x[i]), width - 1)
        ch = 1 if p[i] > 0 else 0
        frames[f, ch, yi, xi] += 1.0

    return frames

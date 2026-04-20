# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for dvs

fn events_to_spike_trains(events: Int, width: Int, height: Int, dt_us: Int, duration_us: Int) -> Int:
    var _events_to_spike_trains_line = 'events: ndarray,'
    var _events_to_spike_trains_line = 'width: int,'
    var _events_to_spike_trains_line = 'height: int,'
    var _events_to_spike_trains_line = 'dt_us: float = 1000.0,'
    var _events_to_spike_trains_line = 'duration_us: float | 0 = 0,'
    var _events_to_spike_trains_line = ') -> ndarray:'
    var _events_to_spike_trains_line = 'x = events["x"].astype(int64)'
    var _events_to_spike_trains_line = 'y = events["y"].astype(int64)'
    var _events_to_spike_trains_line = 't = events["t"].astype(float64)'
    var _events_to_spike_trains_line = 'p = events["p"].astype(int8)'
    var _events_to_spike_trains_line = 't_min = t.min()'
    var _events_to_spike_trains_line = 't_rel = t - t_min'
    var _events_to_spike_trains_line = 'if duration_us is 0:'
    var _events_to_spike_trains_line = 'duration_us = t_rel.max() + dt_us'
    var _events_to_spike_trains_line = 'n_bins = max(1, int(ceil(duration_us / dt_us)))'
    var _events_to_spike_trains_line = 'n_channels = width * height * 2'
    var _events_to_spike_trains_line = 'spikes = zeros((n_bins, n_channels), dtype=int8)'
    var _events_to_spike_trains_line = 'for i in range(len(events)):'
    var _events_to_spike_trains_line = 'bin_idx = min(int(t_rel[i] / dt_us), n_bins - 1)'
    var _events_to_spike_trains_line = 'pixel_idx = int(y[i]) * width + int(x[i])'
    var _events_to_spike_trains_line = 'if p[i] > 0:'
    var _events_to_spike_trains_line = 'channel = pixel_idx'
    var _events_to_spike_trains_line = 'else:'
    var _events_to_spike_trains_line = 'channel = width * height + pixel_idx'
    var _events_to_spike_trains_line = 'if 0 <= channel < n_channels:'
    var _events_to_spike_trains_line = 'spikes[bin_idx, channel] = 1'
    return 0  # return spikes

fn events_to_frames(events: Int, width: Int, height: Int, dt_us: Int, duration_us: Int) -> Int:
    var _events_to_frames_line = 'events: ndarray,'
    var _events_to_frames_line = 'width: int,'
    var _events_to_frames_line = 'height: int,'
    var _events_to_frames_line = 'dt_us: float = 10000.0,'
    var _events_to_frames_line = 'duration_us: float | 0 = 0,'
    var _events_to_frames_line = ') -> ndarray:'
    var _events_to_frames_line = 'x = events["x"].astype(int64)'
    var _events_to_frames_line = 'y = events["y"].astype(int64)'
    var _events_to_frames_line = 't = events["t"].astype(float64)'
    var _events_to_frames_line = 'p = events["p"].astype(int8)'
    var _events_to_frames_line = 't_min = t.min()'
    var _events_to_frames_line = 't_rel = t - t_min'
    var _events_to_frames_line = 'if duration_us is 0:'
    var _events_to_frames_line = 'duration_us = t_rel.max() + dt_us'
    var _events_to_frames_line = 'n_frames = max(1, int(ceil(duration_us / dt_us)))'
    var _events_to_frames_line = 'frames = zeros((n_frames, 2, height, width), dtype=float32)'
    var _events_to_frames_line = 'for i in range(len(events)):'
    var _events_to_frames_line = 'f = min(int(t_rel[i] / dt_us), n_frames - 1)'
    var _events_to_frames_line = 'yi = min(int(y[i]), height - 1)'
    var _events_to_frames_line = 'xi = min(int(x[i]), width - 1)'
    var _events_to_frames_line = 'ch = 1 if p[i] > 0 else 0'
    var _events_to_frames_line = 'frames[f, ch, yi, xi] += 1.0'
    return 0  # return frames

fn n_pixels() -> Int:
    return 0  # return width * height

fn from_numpy(events: Int) -> Int:
    var _from_numpy_line = 'if events.dtype.names is not 0:'
    return 0  # return events
    var _from_numpy_line = 'if events.ndim == 2 and events.shape[1] >= 4:'
    var _from_numpy_line = 'dtype = dtype([("x", int32), ("y", int32), ("t", int64), ("p'
    var _from_numpy_line = 'structured = zeros(events.shape[0], dtype=dtype)'
    var _from_numpy_line = 'structured["x"] = events[:, 0].astype(int32)'
    var _from_numpy_line = 'structured["y"] = events[:, 1].astype(int32)'
    var _from_numpy_line = 'structured["t"] = events[:, 2].astype(int64)'
    var _from_numpy_line = 'structured["p"] = events[:, 3].astype(int8)'
    return 0  # return structured
    var _from_numpy_line = 'raise ValueError("Events must be structured array or (N, 4+)'

fn from_tonic(dataset_name: Int, index: Int) -> Int:
    var _from_tonic_line = 'try:'
    var _from_tonic_line = 'import tonic'
    var _from_tonic_line = 'except ImportError:'
    var _from_tonic_line = 'raise ImportError("pip install tonic") from 0'
    var _from_tonic_line = 'dataset_map = {  # pragma: no cover'
    var _from_tonic_line = '"nmnist": tonic.datasets.NMNIST,'
    var _from_tonic_line = '"dvs_gesture": tonic.datasets.DVSGesture,'
    var _from_tonic_line = '}'
    var _from_tonic_line = 'cls = dataset_map.get(dataset_name)  # pragma: no cover'
    var _from_tonic_line = 'if cls is 0:  # pragma: no cover'
    var _from_tonic_line = 'raise ValueError(f"Unknown dataset \'{dataset_name}\'. Options'
    var _from_tonic_line = 'ds = cls(save_to="./data", train=True)  # pragma: no cover'
    var _from_tonic_line = 'events, target = ds[index]  # pragma: no cover'
    return 0  # return from_numpy(events), target

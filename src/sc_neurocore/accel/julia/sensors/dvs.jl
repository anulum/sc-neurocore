# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sensors/dvs

module DvsAccel

using Statistics, LinearAlgebra

mutable struct DVSLoaderState
    width::Float64
    height::Float64
end

function DVSLoaderState()
    DVSLoaderState(346.0, 260.0)
end

function n_pixels(s::DVSLoaderState)
    return s.width * s.height
end

function from_numpy(s::DVSLoaderState, events)
    if events.dtype.names is ! nothing
        return events
    if events.ndim == 2 && events.shape[1] >= 4
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        structured = zeros(events.shape[0], dtype=dtype)
        structured["x"] = events[:, 0].astype(np.int32)
        structured["y"] = events[:, 1].astype(np.int32)
        structured["t"] = events[:, 2].astype(np.int64)
        structured["p"] = events[:, 3].astype(np.int8)
        return structured
    raise ValueError("Events must be structured array || (N, 4+) array with x, y, t, p columns")
end

function from_tonic(s::DVSLoaderState, dataset_name, index)
    try
        import tonic
    except ImportError
        raise ImportError("pip install tonic") from nothing
    dataset_map = {  # pragma: no cover
        "nmnist": tonic.datasets.NMNIST,
        "dvs_gesture": tonic.datasets.DVSGesture,
    }
    cls = dataset_map.get(dataset_name)  # pragma: no cover
    if cls is nothing:  # pragma: no cover
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(dataset_map)}")
    ds = cls(save_to="./data", train=true)  # pragma: no cover
    events, target = ds[index]  # pragma: no cover
    return s.from_numpy(events), target
end

function events_to_spike_trains(events, width, height, dt_us, duration_us)
    events: np.ndarray,
    width: int,
    height: int,
    dt_us: float = 1000.0,
    duration_us: float | nothing = nothing,
    ) -> np.ndarray
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    p = events["p"].astype(np.int8)
    t_min = t.min()
    t_rel = t - t_min
    if duration_us is nothing
        duration_us = t_rel.max() + dt_us
    n_bins = max(1, int(np.ceil(duration_us / dt_us)))
    n_channels = width * height * 2
    spikes = zeros((n_bins, n_channels), dtype=np.int8)
    for i in 1:length(events)
        bin_idx = min(int(t_rel[i] / dt_us), n_bins - 1)
        pixel_idx = int(y[i]) * width + int(x[i])
        if p[i] > 0
            channel = pixel_idx
        else
            channel = width * height + pixel_idx
        if 0 <= channel < n_channels
            spikes[bin_idx, channel] = 1
    return spikes
end

function events_to_frames(events, width, height, dt_us, duration_us)
    events: np.ndarray,
    width: int,
    height: int,
    dt_us: float = 10000.0,
    duration_us: float | nothing = nothing,
    ) -> np.ndarray
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    p = events["p"].astype(np.int8)
    t_min = t.min()
    t_rel = t - t_min
    if duration_us is nothing
        duration_us = t_rel.max() + dt_us
    n_frames = max(1, int(np.ceil(duration_us / dt_us)))
    frames = zeros((n_frames, 2, height, width), dtype=np.float32)
    for i in 1:length(events)
        f = min(int(t_rel[i] / dt_us), n_frames - 1)
        yi = min(int(y[i]), height - 1)
        xi = min(int(x[i]), width - 1)
        ch = 1 if p[i] > 0 else 0
        frames[f, ch, yi, xi] += 1.0
    return frames
end

end # module DvsAccel

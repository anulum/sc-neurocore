# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.sensors (DVS event camera pipeline)

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.sensors import DVSLoader, events_to_spike_trains, events_to_frames


def _make_events(
    n: int = 100, width: int = 8, height: int = 6, seed: int = 42
) -> npt.NDArray[np.void]:
    rng = np.random.RandomState(seed)
    dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
    events = np.zeros(n, dtype=dtype)
    events["x"] = rng.randint(0, width, n)
    events["y"] = rng.randint(0, height, n)
    events["t"] = np.sort(rng.randint(0, 100000, n))
    events["p"] = rng.choice([0, 1], n)
    return events


class TestDVSLoader:
    def test_n_pixels(self) -> None:
        loader = DVSLoader(width=8, height=6)
        assert loader.n_pixels == 48

    def test_default_dims(self) -> None:
        loader = DVSLoader()
        assert loader.width == 346
        assert loader.height == 260

    def test_from_numpy_structured(self) -> None:
        loader = DVSLoader(width=8, height=6)
        events = _make_events()
        result = loader.from_numpy(events)
        assert result is events  # passthrough for structured

    def test_from_numpy_2d_array(self) -> None:
        loader = DVSLoader(width=8, height=6)
        raw = np.array(
            [
                [3, 2, 1000, 1],
                [5, 4, 2000, 0],
                [1, 0, 3000, 1],
            ],
            dtype=np.float64,
        )
        result = loader.from_numpy(raw)
        assert result.dtype.names is not None
        assert result["x"][0] == 3
        assert result["y"][1] == 4
        assert result["p"][2] == 1

    def test_from_numpy_invalid(self) -> None:
        loader = DVSLoader()
        with pytest.raises(ValueError, match="must be structured"):
            loader.from_numpy(np.array([1, 2, 3]))

    def test_from_tonic_import_error(self) -> None:
        loader = DVSLoader()
        with pytest.raises(ImportError, match="pip install tonic"):
            loader.from_tonic("nmnist")


class TestEventsToSpikeTrains:
    def test_basic_shape(self) -> None:
        events = _make_events(n=50, width=4, height=3)
        spikes = events_to_spike_trains(events, width=4, height=3, dt_us=10000.0)
        n_channels = 4 * 3 * 2  # ON + OFF
        assert spikes.shape[1] == n_channels
        assert spikes.shape[0] >= 1

    def test_binary_output(self) -> None:
        events = _make_events()
        spikes = events_to_spike_trains(events, width=8, height=6, dt_us=10000.0)
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_explicit_duration(self) -> None:
        events = _make_events()
        spikes = events_to_spike_trains(
            events,
            width=8,
            height=6,
            dt_us=10000.0,
            duration_us=50000.0,
        )
        assert spikes.shape[0] == 5

    def test_on_off_channels(self) -> None:
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(2, dtype=dtype)
        events[0] = (0, 0, 0, 1)  # ON event at pixel (0,0)
        events[1] = (1, 0, 0, 0)  # OFF event at pixel (1,0)
        spikes = events_to_spike_trains(events, width=2, height=1, dt_us=10000.0)
        # ON channel for pixel 0 = index 0, OFF channel for pixel 1 = index 2+1=3
        assert spikes[0, 0] == 1  # ON pixel 0
        assert spikes[0, 3] == 1  # OFF pixel 1

    def test_empty_after_filtering(self) -> None:
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(1, dtype=dtype)
        events[0] = (0, 0, 500, 1)
        spikes = events_to_spike_trains(events, width=2, height=2, dt_us=1000.0)
        assert spikes.shape[0] >= 1


class TestEventsToFrames:
    def test_basic_shape(self) -> None:
        events = _make_events(n=50, width=4, height=3)
        frames = events_to_frames(events, width=4, height=3, dt_us=10000.0)
        assert frames.ndim == 4
        assert frames.shape[1] == 2  # ON and OFF channels
        assert frames.shape[2] == 3  # height
        assert frames.shape[3] == 4  # width

    def test_accumulates_counts(self) -> None:
        dtype = np.dtype([("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8)])
        events = np.zeros(3, dtype=dtype)
        events[0] = (0, 0, 0, 1)
        events[1] = (0, 0, 500, 1)
        events[2] = (0, 0, 900, 1)
        frames = events_to_frames(events, width=2, height=2, dt_us=2000.0)
        # All 3 events in first frame, ON channel
        assert frames[0, 1, 0, 0] == 3.0

    def test_explicit_duration(self) -> None:
        events = _make_events()
        frames = events_to_frames(
            events,
            width=8,
            height=6,
            dt_us=25000.0,
            duration_us=100000.0,
        )
        assert frames.shape[0] == 4

    def test_float32_dtype(self) -> None:
        events = _make_events()
        frames = events_to_frames(events, width=8, height=6)
        assert frames.dtype == np.float32

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sensors_dvs.py

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


__all__ = [
    "np",
    "npt",
    "pytest",
    "DVSLoader",
    "events_to_spike_trains",
    "events_to_frames",
    "_make_events",
]

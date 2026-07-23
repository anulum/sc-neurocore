# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sensor_fusion.py

from __future__ import annotations

import numpy as np
from sc_neurocore.fusion.sensor_fusion import (
    BitstreamDecorrelator,
    CochleaAdapter,
    CrossModalAttention,
    DVSAdapter,
    EventStream,
    FusionEnergyEstimator,
    FusionVerilogEmitter,
    HDCBinding,
    IMUAdapter,
    SensorFusionLayer,
    SensorModality,
    TactileAdapter,
    TemporalAligner,
)
def _make_stream(
    modality: SensorModality,
    n_events: int = 100,
    seed: int = 0,
) -> EventStream:
    rng = np.random.default_rng(seed)
    return EventStream(
        modality=modality,
        timestamps=np.sort(rng.integers(0, 1_000_000, n_events)).astype(np.float64),
        addresses=rng.integers(0, 64, n_events),
        polarities=rng.choice([-1, 1], n_events),
    )

__all__ = ['np', 'BitstreamDecorrelator', 'CochleaAdapter', 'CrossModalAttention', 'DVSAdapter', 'EventStream', 'FusionEnergyEstimator', 'FusionVerilogEmitter', 'HDCBinding', 'IMUAdapter', 'SensorFusionLayer', 'SensorModality', 'TactileAdapter', 'TemporalAligner', '_make_stream', '__all__']

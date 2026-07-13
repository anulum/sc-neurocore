# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary fitness extraction from MEA activity

"""Evolutionary fitness extraction from MEA activity."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .bioware_contracts import DetectedSpike
from .bioware_validation import require_nonnegative


def mea_fitness_hook(
    detected_spikes: List[DetectedSpike],
    target_rate: float = 10.0,
    *,
    duration_s: Optional[float] = None,
    stimulus_time_s: Optional[float] = None,
    measured_latency_ms: Optional[float] = None,
) -> Dict[str, float]:
    """Organism fitness metrics derived from MEA response dynamics.

    Designed to plug into the evo_substrate
    ``ReplicationEngine(metrics_fn=mea_fitness_hook)`` — returns the
    ``{"accuracy", "energy_mw", "latency_ms"}`` triple the engine scores.

    Accuracy is a bounded distance to the target mean per-channel firing
    rate when ``duration_s`` is supplied, or to the legacy per-channel
    spike count when it is omitted. The legacy ``energy_mw`` key is a
    dimensionless optimisation proxy equal to ``0.5 * spike_count``; it is
    not a measured power or energy quantity. ``latency_ms`` is either a
    caller-supplied closed-loop measurement, the first response latency after
    ``stimulus_time_s``, or the first spike timestamp relative to frame start.
    """
    require_nonnegative(target_rate, "target_rate")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("duration_s must be finite and > 0 when provided")
    if stimulus_time_s is not None and not math.isfinite(stimulus_time_s):
        raise ValueError("stimulus_time_s must be finite when provided")
    if measured_latency_ms is not None:
        if not math.isfinite(measured_latency_ms) or measured_latency_ms < 0.0:
            raise ValueError("measured_latency_ms must be finite and >= 0 when provided")

    if not detected_spikes:
        latency_ms = _mea_response_latency_ms(
            detected_spikes,
            stimulus_time_s=stimulus_time_s,
            measured_latency_ms=measured_latency_ms,
        )
        return {"accuracy": 0.1, "energy_mw": 0.0, "latency_ms": latency_ms}

    counts: Dict[int, float] = {}
    for s in detected_spikes:
        counts[s.channel] = counts.get(s.channel, 0.0) + 1.0

    per_channel_activity = np.array(list(counts.values()), dtype=float)
    if duration_s is not None:
        per_channel_activity = per_channel_activity / duration_s
    mean_rate = float(np.mean(per_channel_activity)) if per_channel_activity.size else 0.0

    # Normalised distance to target rate → accuracy ∈ [0.1, 0.99].
    if target_rate > 0.0:
        accuracy = 1.0 - min(1.0, abs(mean_rate - target_rate) / target_rate)
    else:
        accuracy = 0.1

    latency_ms = _mea_response_latency_ms(
        detected_spikes,
        stimulus_time_s=stimulus_time_s,
        measured_latency_ms=measured_latency_ms,
    )
    return {
        "accuracy": float(np.clip(accuracy, 0.1, 0.99)),
        "energy_mw": float(len(detected_spikes) * 0.5),
        "latency_ms": latency_ms,
    }


def _mea_response_latency_ms(
    detected_spikes: List[DetectedSpike],
    *,
    stimulus_time_s: Optional[float],
    measured_latency_ms: Optional[float],
) -> float:
    if measured_latency_ms is not None:
        return float(measured_latency_ms)

    timestamps = np.array([s.timestamp_s for s in detected_spikes], dtype=float)
    if timestamps.size == 0:
        return 0.0
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("detected spike timestamps must be finite")

    if stimulus_time_s is not None:
        responses = timestamps[timestamps >= stimulus_time_s]
        if responses.size == 0:
            return 0.0
        return float((np.min(responses) - stimulus_time_s) * 1000.0)

    return float(max(0.0, np.min(timestamps)) * 1000.0)

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated deterministic Bioware benchmark probe

"""Emit one cold-process Bioware fidelity and timing sample."""

from __future__ import annotations

import hashlib
import json
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np


def _voltage() -> np.ndarray[Any, Any]:
    """Build a deterministic 50 ms, eight-channel MEA frame."""
    voltage = np.random.default_rng(20260713).normal(0.0, 3.0, size=(1000, 8))
    for sample, channel, amplitude in (
        (50, 0, -80.0),
        (125, 3, -65.0),
        (250, 0, -75.0),
        (475, 5, -70.0),
        (700, 3, -82.0),
        (925, 7, -68.0),
    ):
        voltage[sample, channel] = amplitude
    return voltage


def _pipeline() -> dict[str, Any]:
    """Run the maintained deterministic Python pipeline once."""
    from sc_neurocore.bioware.bioware import (
        AERToSCConverter,
        CultureHealth,
        MEAConfig,
        MEAToAERTranscoder,
        SCToOptoEncoder,
        SpikeDetector,
        extract_lfp_power,
        mea_fitness_hook,
    )

    config = MEAConfig(
        num_channels=8,
        sample_rate_hz=20_000.0,
        spike_threshold_sigma=4.0,
    )
    voltage = _voltage()
    spikes = SpikeDetector(config=config).detect(voltage)
    events = MEAToAERTranscoder(hw_clock_hz=1e6).transcode(spikes)
    bitstreams = AERToSCConverter(
        window_ticks=0x10000,
        bitstream_length=512,
        num_neurons=config.num_channels,
        lfsr_seed=0xACE1,
    ).convert(events)
    pulses = SCToOptoEncoder(
        max_intensity_mw_mm2=5.0,
        max_total_power_mw=50.0,
    ).encode(bitstreams)
    counts = np.zeros(config.num_channels)
    for spike in spikes:
        counts[spike.channel] += 1
    health = CultureHealth(min_active_channels=3).assess(counts, duration_s=0.05)
    lfp = extract_lfp_power(voltage, config.sample_rate_hz)
    fitness = mea_fitness_hook(spikes, duration_s=0.05, target_rate=20.0)
    return {
        "spikes": [
            {
                "channel": spike.channel,
                "timestamp_s": spike.timestamp_s,
                "amplitude_uv": spike.amplitude_uv,
                "unit_id": spike.unit_id,
                "waveform_sha256": hashlib.sha256(
                    np.ascontiguousarray(spike.waveform).tobytes()
                ).hexdigest(),
            }
            for spike in spikes
        ],
        "events": [
            {
                "neuron_id": event.neuron_id,
                "timestamp": event.timestamp,
                "valid": event.valid,
                "weight": event.weight,
            }
            for event in events
        ],
        "bitstreams": {
            str(neuron_id): np.ascontiguousarray(bits).tobytes().hex()
            for neuron_id, bits in sorted(bitstreams.items())
        },
        "pulses": [
            {
                "channel": pulse.channel,
                "onset_ms": pulse.onset_ms,
                "duration_ms": pulse.duration_ms,
                "intensity_mw_mm2": pulse.intensity_mw_mm2,
                "wavelength_nm": pulse.wavelength_nm,
            }
            for pulse in pulses
        ],
        "health": health,
        "lfp": {name: values.tolist() for name, values in sorted(lfp.items())},
        "fitness": fitness,
    }


def _canonical_payload(result: dict[str, Any]) -> bytes:
    """Serialise all maintained public-output contracts deterministically."""
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> int:
    import_started = time.perf_counter_ns()
    from sc_neurocore.bioware import bioware as historical

    import_ns = time.perf_counter_ns() - import_started
    if not hasattr(historical, "BioHybridSession"):
        raise RuntimeError("historical Bioware module has no session surface")
    samples: list[int] = []
    result: dict[str, Any] = {}
    for _ in range(12):
        started = time.perf_counter_ns()
        result = _pipeline()
        samples.append(time.perf_counter_ns() - started)
    payload = _canonical_payload(result)
    maximum_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        maximum_rss_kib //= 1024
    print(
        json.dumps(
            {
                "import_ns": import_ns,
                "pipeline_ns": int(statistics.median(samples)),
                "max_rss_kib": maximum_rss_kib,
                "canonical_sha256": hashlib.sha256(payload).hexdigest(),
                "canonical_bytes": len(payload),
                "spike_count": len(result["spikes"]),
                "aer_event_count": len(result["events"]),
                "bitstream_count": len(result["bitstreams"]),
                "opto_pulse_count": len(result["pulses"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

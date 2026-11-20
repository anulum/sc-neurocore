# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recorded event and band-power stream measurements

"""Compare complete spike-event streams on original mono and multichannel recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any, NamedTuple

import numpy as np

from bench_isi_codec import _module_hash
from bench_predictive_codec import _host
from bench_waveform_container import _wav
from bench_waveform_stream_fixture import SAMPLE_RATE, read_recording
from sc_neurocore.spike_codec import SpikeEventDecoder, SpikeEventEncoder, SpikeEventWindow

_BACKENDS = ("python", "rust")
_SOURCES = (
    "Cargo.lock",
    "engine/Cargo.toml",
    "engine/src/spike_events.rs",
    "engine/src/waveform_stream.rs",
    "engine/src/bindings/spike_events.rs",
    "bridge/sc_neurocore_engine/__init__.py",
    "src/sc_neurocore/spike_codec/spike_events.py",
    "src/sc_neurocore/spike_codec/waveform_stream.py",
    "benchmarks/bench_spike_event_stream.py",
    "benchmarks/bench_waveform_stream_fixture.py",
    "benchmarks/bench_waveform_container.py",
)


class Layout(NamedTuple):
    """Event precision, packet span, detector bin and channel groups."""

    timing: str
    rows: int
    bin_samples: int
    groups: int


class Run(NamedTuple):
    """Recorded packets, decoded windows and normalized measurements."""

    packets: list[bytes]
    windows: list[SpikeEventWindow]
    timings: dict[str, int]


def _pair(
    channels: int, rate: int, layout: Layout, backend: str
) -> tuple[SpikeEventEncoder, SpikeEventDecoder]:
    """Configure both sides for the original acquisition's sample rate."""
    low_rate = rate <= 1000
    encoder = SpikeEventEncoder(
        channels,
        rate,
        timing=layout.timing,
        groups=layout.groups,
        bin_samples=layout.bin_samples,
        power_bins=5,
        highpass_hz=3.0 if low_rate else 250.0,
        lowpass_hz=35.0 if low_rate else 1000.0,
        backend=backend,
    )
    decoder = SpikeEventDecoder(
        channels,
        layout.bin_samples,
        timing=layout.timing,
        groups=layout.groups,
        power_bins=5,
        backend=backend,
    )
    return encoder, decoder


def _run(values: np.ndarray[Any, Any], rate: int, layout: Layout, backend: str) -> Run:
    """Time detector and encoder work and the decoder on the actual packet boundaries."""
    start = perf_counter_ns()
    encoder, decoder = _pair(values.shape[1], rate, layout, backend)
    setup_ns = perf_counter_ns() - start
    packets = []
    windows = []
    latencies = []
    encode_ns = decode_ns = 0
    for pos in range(0, len(values), layout.rows):
        chunk = values[pos : pos + layout.rows]
        start = perf_counter_ns()
        packet = encoder.encode(chunk)
        encoded = perf_counter_ns()
        window = decoder.decode(packet, len(chunk))
        done = perf_counter_ns()
        encode_ns += encoded - start
        decode_ns += done - encoded
        latencies.append(done - start)
        packets.append(packet)
        windows.append(window)
    return Run(
        packets,
        windows,
        {
            "setup_ns": setup_ns,
            "encode_ns": encode_ns,
            "decode_ns": decode_ns,
            "total_ns": setup_ns + encode_ns + decode_ns,
            "p95_packet_ns": sorted(latencies)[math.ceil(len(latencies) * 0.95) - 1],
        },
    )


def _same_windows(expected: list[SpikeEventWindow], actual: list[SpikeEventWindow]) -> None:
    """Require exact equality of the complete event, offset and power outputs."""
    for left, right in zip(expected, actual, strict=True):
        if (left.first_bin, left.first_period) != (right.first_bin, right.first_period):
            raise RuntimeError("decoded event or power indices differ")
        for a, b in (
            (left.events, right.events),
            (left.offsets, right.offsets),
            (left.power, right.power),
        ):
            equal = a is b if a is None or b is None else np.array_equal(a, b)
            if not equal:
                raise RuntimeError("decoded event, offset or power values differ")


def _case(
    values: np.ndarray[Any, Any], rate: int, layout: Layout, repetitions: int
) -> dict[str, object]:
    """Run alternating backend pairs and cross-decode every measured packet."""
    runs: dict[str, list[dict[str, int]]] = {backend: [] for backend in _BACKENDS}
    expected = _run(values, rate, layout, "python")
    encoded: dict[str, list[bytes]] = {}
    for repeat in range(repetitions):
        for backend in _BACKENDS if repeat % 2 == 0 else _BACKENDS[::-1]:
            actual = _run(values, rate, layout, backend)
            if actual.packets != expected.packets:
                raise RuntimeError("Python and Rust spike-event packet bytes differ")
            _same_windows(expected.windows, actual.windows)
            runs[backend].append(actual.timings)
            encoded[backend] = actual.packets
    for backend, peer in zip(_BACKENDS, _BACKENDS[::-1], strict=True):
        _, decoder = _pair(values.shape[1], rate, layout, backend)
        restored = [
            decoder.decode(packet, min(layout.rows, len(values) - pos * layout.rows))
            for pos, packet in enumerate(encoded[peer])
        ]
        _same_windows(expected.windows, restored)
    bins = sum(len(window.events) for window in expected.windows)
    periods = sum(len(window.power) for window in expected.windows)
    if bins != len(values) // layout.bin_samples or periods != len(values) // (
        layout.bin_samples * 5
    ):
        raise RuntimeError("decoded partial windows lost a completed bin or power period")
    events = sum(int(window.events.sum()) for window in expected.windows)
    packet_bytes = sum(map(len, expected.packets))
    return {
        **layout._asdict(),
        "packet_count": len(expected.packets),
        "packet_bytes": packet_bytes,
        "packet_sha256": hashlib.sha256(
            b"".join(len(packet).to_bytes(8, "big") + packet for packet in expected.packets)
        ).hexdigest(),
        "events": events,
        "events_per_second": events * rate / len(values),
        "bits_per_second": packet_bytes * 8 * rate / len(values),
        "power_periods": periods,
        "runs": runs,
        "median_total_ns": {
            backend: statistics.median(run["total_ns"] for run in samples)
            for backend, samples in runs.items()
        },
        "median_p95_packet_ns": {
            backend: statistics.median(run["p95_packet_ns"] for run in samples)
            for backend, samples in runs.items()
        },
    }


def main() -> None:
    """Read original recordings and write the paired benchmark receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--edf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    wav, wav_rate = _wav(args.wav)
    recordings = []
    for name, path, values, rate, bin_samples in (
        ("mono challenge", args.wav, wav, wav_rate, round(wav_rate / 1000)),
        ("64-channel EEG", args.edf, read_recording(args.edf), SAMPLE_RATE, 4),
    ):
        if not len(values):
            parser.error("recordings must contain at least one sample")
        layouts = (
            Layout("bin", max(1, round(rate * 0.040)) + 1, bin_samples, 1),
            Layout("sample", max(1, round(rate * 0.065)) + 1, bin_samples, min(8, values.shape[1])),
        )
        recordings.append(
            {
                "recording": name,
                "filename": path.name,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": values.shape[0],
                "channels": values.shape[1],
                "sample_rate_hz": rate,
                "results": [_case(values, rate, layout, args.repetitions) for layout in layouts],
            }
        )
    root = Path(__file__).resolve().parents[1]
    result = {
        "schema": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host": _host(),
        "rayon_num_threads": os.getenv("RAYON_NUM_THREADS"),
        "installed_sha256": {
            name: _module_hash(name)
            for name in (
                "sc_neurocore.spike_codec.spike_events",
                "sc_neurocore_engine.sc_neurocore_engine",
            )
        },
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _SOURCES
        },
        "repetitions": args.repetitions,
        "recordings": recordings,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

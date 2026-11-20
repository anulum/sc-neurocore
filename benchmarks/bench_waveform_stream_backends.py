# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end waveform stream backend benchmark

"""Compare installed Python, Rust CPU and device encoding with CPU decode on real EEG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from bench_isi_codec import _module_hash
from bench_predictive_codec import _host
from bench_waveform_stream_fixture import SAMPLE_RATE, SHA256, SOURCE, read_recording
from sc_neurocore.spike_codec import LosslessStreamDecoder, LosslessStreamEncoder

_CASES = ((1, 1, 17), (8, 3, 173), (8, 8, 173), (64, 64, 173))
_SOURCES = (
    "Cargo.lock",
    "benchmarks/bench_isi_codec.py",
    "benchmarks/bench_predictive_codec.py",
    "benchmarks/bench_waveform_stream_backends.py",
    "benchmarks/bench_waveform_stream_fixture.py",
    "bridge/pyproject.toml",
    "engine/Cargo.toml",
    "engine/rust-toolchain.toml",
    "src/sc_neurocore/spike_codec/waveform_stream.py",
    "engine/src/waveform_stream.rs",
    "engine/src/waveform_stream_gpu.rs",
    "engine/src/waveform_stream_gpu.wgsl",
    "engine/src/bindings/waveform_stream.rs",
)


def _run(
    backend: str, values: np.ndarray[Any, Any], groups: int, rows: int
) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """Time construction and sequential packet delivery, verifying every output after timing."""
    channels = values.shape[1]
    windows = tuple(values[start : start + rows] for start in range(0, len(values), rows))
    start = perf_counter_ns()
    encoder = LosslessStreamEncoder(
        channels, 16, groups, backend, gpu_max_samples=rows if backend == "gpu" else None
    )
    decoder = LosslessStreamDecoder(channels, 16, groups, "rust" if backend == "gpu" else backend)
    setup_ns = perf_counter_ns() - start
    if encoder.backend != backend:
        raise RuntimeError(f"unexpected backend switch from {backend} to {encoder.backend}")

    encode_ns: list[int] = []
    decode_ns: list[int] = []
    packets: list[bytes] = []
    for window in windows:
        start = perf_counter_ns()
        packet = encoder.encode(window)
        encoded = perf_counter_ns()
        restored = decoder.decode(packet, len(window))
        decoded = perf_counter_ns()
        encode_ns.append(encoded - start)
        decode_ns.append(decoded - encoded)
        packets.append(packet)
        if not np.array_equal(restored, window):
            raise RuntimeError("decoded samples differ from the recording")
    return {
        "setup_ns": setup_ns,
        "encode_ns": encode_ns,
        "decode_ns": decode_ns,
        "total_ns": setup_ns + sum(encode_ns) + sum(decode_ns),
        "device": encoder.device,
        "decoder": decoder.backend,
    }, tuple(packets)


def _case(
    values: np.ndarray[Any, Any],
    groups: int,
    rows: int,
    repetitions: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Rotate lane order and match every encoded packet across languages and devices."""
    backends = ["python", "rust"]
    if use_gpu and groups == values.shape[1]:
        backends.append("gpu")
    runs: dict[str, list[dict[str, Any]]] = {backend: [] for backend in backends}
    reference: tuple[bytes, ...] | None = None
    for repeat in range(repetitions):
        offset = repeat % len(backends)
        for backend in backends[offset:] + backends[:offset]:
            metrics, packets = _run(backend, values, groups, rows)
            if reference is None:
                reference = packets
            elif packets != reference:
                raise RuntimeError(f"{backend} emitted different packets")
            runs[backend].append(metrics)
    assert reference is not None
    return {
        "channels": values.shape[1],
        "groups": groups,
        "adc_bits": 16,
        "window_rows": rows,
        "tail_rows": len(values) % rows or rows,
        "packets": len(reference),
        "rows": len(values),
        "input_bytes": values.nbytes,
        "packet_bytes": sum(map(len, reference)),
        "packet_sha256": hashlib.sha256(b"".join(reference)).hexdigest(),
        "median_total_ns": {
            backend: statistics.median(run["total_ns"] for run in samples)
            for backend, samples in runs.items()
        },
        "runs": runs,
    }


def main() -> None:
    """Measure the pinned recording and write raw timings, provenance and exact packet hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="original PhysioNet S001R01.edf")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--gpu-driver", help="record the selected graphics driver and version")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.gpu and not args.gpu_driver:
        parser.error("--gpu requires --gpu-driver from the executing machine")
    values = read_recording(args.recording)
    root = Path(__file__).resolve().parent.parent
    results = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host": _host(),
        "rayon_num_threads": os.getenv("RAYON_NUM_THREADS"),
        "gpu_driver": args.gpu_driver,
        "recording": {"url": SOURCE, "sha256": SHA256, "sample_rate_hz": SAMPLE_RATE},
        "method": "constructor setup plus encode and CPU decode for every packet; rotating lanes",
        "repetitions": args.repetitions,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in _SOURCES
        },
        "installed_sha256": {
            module: _module_hash(module)
            for module in (
                "sc_neurocore.spike_codec.waveform_stream",
                "sc_neurocore_engine.sc_neurocore_engine",
            )
        },
        "cases": [
            _case(
                np.ascontiguousarray(values[:, :channels]), groups, rows, args.repetitions, args.gpu
            )
            for channels, groups, rows in _CASES
        ],
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

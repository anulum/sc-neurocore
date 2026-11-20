# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Whole-container waveform codec measurements

"""Compare complete WFC2 encoding, decoding and reconstruction on recorded samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import wave
import zlib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

from bench_isi_codec import _module_hash
from bench_predictive_codec import _host
from bench_waveform_stream_fixture import SAMPLE_RATE, read_recording
from sc_neurocore.spike_codec import WaveformCodec, WaveformDecodeResult

_MODES = ("spike", "waveform", "full", "lossless", "lossless_rice")
_BACKENDS = ("python", "rust")
_SOURCES = (
    "Cargo.lock",
    "engine/Cargo.toml",
    "engine/src/waveform_codec.rs",
    "engine/src/waveform_stream.rs",
    "engine/src/codecs/isi.rs",
    "engine/src/bindings/waveform_container.rs",
    "src/sc_neurocore/spike_codec/_waveform_container.py",
    "src/sc_neurocore/spike_codec/waveform_codec.py",
    "benchmarks/bench_waveform_container.py",
    "benchmarks/bench_waveform_stream_fixture.py",
)


def _wav(path: Path) -> tuple[np.ndarray[Any, Any], int]:
    """Read a PCM16 WAV as channel-interleaved rows with its acquisition rate."""
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("the reference recording must be PCM16")
        frames = reader.getnframes()
        channels = reader.getnchannels()
        rate = reader.getframerate()
        samples = np.frombuffer(reader.readframes(frames), dtype="<i2").reshape(frames, channels)
    return np.asarray(samples, dtype=np.int16), rate


def _shape_reference(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Calculate threshold crossings and refractory filtering directly on ADC values."""
    values = values.astype(np.float32)
    noise = np.maximum(np.median(np.abs(values), axis=0) / np.float32(0.6745), np.float32(1e-6))
    eligible = (values[1:] < -np.float32(4.5) * noise) & (values[1:] < values[:-1])
    accepted = np.zeros(values.shape, dtype=np.int8)
    refractory = 24
    for channel, column in enumerate(eligible.T):
        last = -refractory - 1
        for row_index in np.flatnonzero(column):
            row = int(row_index) + 1
            if row - last > refractory:
                accepted[row, channel] = 1
                last = row
    return accepted


def _quality(
    values: np.ndarray[Any, Any],
    decoded: WaveformDecodeResult,
    reference_raster: np.ndarray[Any, Any],
) -> dict[str, float | int | None]:
    """Report ADC reconstruction error and agreement with the defined detector."""
    raster = decoded.spike_raster
    detected = None if raster is None else int(raster.sum())
    if raster is not None and not np.array_equal(raster, reference_raster):
        raise RuntimeError("WFC2 raster differs from the reference detector")
    if decoded.waveform is None:
        return {"detected_events": detected, "event_precision": 1.0, "event_recall": 1.0}
    restored = decoded.waveform
    if decoded.lossless_waveform and not np.array_equal(restored, values):
        raise RuntimeError("WFC2 lossless round trip changed a sample")
    signal = values.astype(np.float64)
    error = restored - signal
    squared_error = float(np.mean(np.square(error)))
    power = float(np.mean(np.square(signal)))
    return {
        "detected_events": detected,
        "event_precision": None if raster is None else 1.0,
        "event_recall": None if raster is None else 1.0,
        "rmse_adc_units": math.sqrt(squared_error),
        "max_abs_error_adc_units": float(np.max(np.abs(error))),
        "snr_db": 10 * math.log10(power / squared_error) if squared_error and power else None,
    }


def _equal_arrays(a: np.ndarray[Any, Any] | None, b: np.ndarray[Any, Any] | None) -> bool:
    """Compare two arrays or confirm both modes omit the same result."""
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(a, b))


def _case(values: np.ndarray[Any, Any], mode: str, repetitions: int) -> dict[str, Any]:
    """Rotate backend order and time fresh encoding and decoding through the installed API."""
    runs: dict[str, list[dict[str, int]]] = {backend: [] for backend in _BACKENDS}
    packets: dict[str, bytes] = {}
    for repeat in range(repetitions):
        order = _BACKENDS if repeat % 2 == 0 else _BACKENDS[::-1]
        for backend in order:
            start = perf_counter_ns()
            codec = WaveformCodec(mode=mode, backend=backend)
            initialized = perf_counter_ns()
            packet, metadata = codec.compress(values)
            encoded = perf_counter_ns()
            restored = WaveformCodec.decompress(packet, backend=backend)
            decoded = perf_counter_ns()
            if metadata.compressed_bytes != len(packet) or restored.mode != mode:
                raise RuntimeError("WFC2 metadata differs from the measured packet")
            if backend in packets and packets[backend] != packet:
                raise RuntimeError("the encoder produced a nondeterministic packet")
            packets[backend] = packet
            runs[backend].append(
                {
                    "setup_ns": initialized - start,
                    "encode_ns": encoded - initialized,
                    "decode_ns": decoded - encoded,
                    "total_ns": decoded - start,
                }
            )
    if packets["python"] != packets["rust"]:
        raise RuntimeError("the reference and native encoders disagree on the complete packet")
    python = WaveformCodec.decompress(packets["rust"], backend="python")
    rust = WaveformCodec.decompress(packets["python"], backend="rust")
    if not _equal_arrays(python.spike_raster, rust.spike_raster) or not _equal_arrays(
        python.waveform, rust.waveform
    ):
        raise RuntimeError("cross-decoded WFC2 arrays differ")
    return {
        "mode": mode,
        "rows": values.shape[0],
        "channels": values.shape[1],
        "packet_bytes": len(packets["rust"]),
        "packet_sha256": hashlib.sha256(packets["rust"]).hexdigest(),
        "quality": _quality(values, rust, _shape_reference(values)),
        "total_median_ns": {
            backend: statistics.median(run["total_ns"] for run in samples)
            for backend, samples in runs.items()
        },
        "runs": runs,
    }


def _version(distribution: str) -> str | None:
    """Return the version of an installed optional codec dependency."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def main() -> None:
    """Measure all modes on a mono challenge recording and a multichannel EEG acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--edf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    raw, wav_rate = _wav(args.wav)
    eeg = read_recording(args.edf)
    recordings = []
    for label, path, values, rate in (
        ("challenge recording", args.wav, raw, wav_rate),
        ("simultaneous EEG", args.edf, eeg, SAMPLE_RATE),
    ):
        recordings.append(
            {
                "recording": label,
                "filename": path.name,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_bytes": values.nbytes,
                "sample_rate_hz": rate,
                "results": [_case(values, mode, args.repetitions) for mode in _MODES],
            }
        )
    root = Path(__file__).resolve().parent.parent
    sources = [
        *_SOURCES,
        *(
            path.relative_to(root).as_posix()
            for path in sorted((root / "engine/src/codecs/waveform_container").glob("*.rs"))
        ),
    ]
    output = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host": _host(),
        "rayon_num_threads": os.getenv("RAYON_NUM_THREADS"),
        "python_zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "optional_distributions": {
            "zstandard": _version("zstandard"),
            "PyWavelets": _version("PyWavelets"),
        },
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
        "installed_sha256": {
            name: _module_hash(name)
            for name in (
                "sc_neurocore.spike_codec.waveform_codec",
                "sc_neurocore_engine.sc_neurocore_engine",
            )
        },
        "repetitions": args.repetitions,
        "recordings": recordings,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

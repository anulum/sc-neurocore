# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive packet codec benchmark

"""Compare complete Python and Rust predictive paths on identical binary rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from bench_isi_codec import Raster, _expected, _hash, _module_hash, _shd, _synthetic
from isi_codec_timing import OPERATIONS, _peak_rss, summarize_samples
from sc_neurocore.spike_codec import PredictiveSpikeCodec

_PREDICTORS = ("ema", "lfsr", "context", "world_model")
_BACKENDS = ("python", "rust")


@dataclass(frozen=True)
class _CodecSettings:
    """Hold the same predictor and timing precision for both backends."""

    predictor: str
    precision: int

    def build(self, backend: str) -> PredictiveSpikeCodec:
        """Select the same predictor and packet format for either backend."""
        return PredictiveSpikeCodec(
            predictor=self.predictor,
            base_mode="lossless" if self.precision == 1 else "lossy",
            timing_precision=self.precision,
            backend=backend,
        )


def _reconstruction(raster: Raster, precision: int) -> Raster:
    """Build the independent expected original-sized decode grid."""
    if precision == 1:
        return raster
    bins = _expected(raster, precision)
    expected = np.zeros_like(raster)
    expected[: len(bins)] = bins
    return expected


def _validate(
    codecs: tuple[PredictiveSpikeCodec, PredictiveSpikeCodec],
    raster: Raster,
    expected: Raster,
) -> bytes:
    """Require exact packets, statistics and cross-decoded rasters."""
    first, first_stats = codecs[0].compress(raster)
    second, second_stats = codecs[1].compress(raster)
    if first != second or first_stats != second_stats:
        raise ValueError("Python and Rust predictive packets or statistics differ")
    for reader, packet in zip(codecs, (second, first), strict=True):
        decoded = reader.decompress(packet, *raster.shape)
        if decoded.dtype != expected.dtype or not np.array_equal(decoded, expected):
            raise ValueError("predictive cross-decoding changed the expected raster")
    return first


def _warm_samples(
    codecs: tuple[PredictiveSpikeCodec, PredictiveSpikeCodec],
    raster: Raster,
    packet: bytes,
    warmups: int,
    repetitions: int,
) -> dict[str, dict[str, list[int]]]:
    """Measure resident public calls and alternate the backend order."""
    samples: dict[str, dict[str, list[int]]] = {
        backend: {op: [] for op in OPERATIONS} for backend in _BACKENDS
    }
    for iteration in range(warmups + repetitions):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            codec = codecs[index]
            start = perf_counter_ns()
            codec.compress(raster)
            encode_ns = perf_counter_ns() - start
            start = perf_counter_ns()
            codec.decompress(packet, *raster.shape)
            decode_ns = perf_counter_ns() - start
            start = perf_counter_ns()
            generated, _ = codec.compress(raster)
            codec.decompress(generated, *raster.shape)
            roundtrip_ns = perf_counter_ns() - start
            if iteration >= warmups:
                series = samples[_BACKENDS[index]]
                for op, duration in zip(
                    OPERATIONS, (encode_ns, decode_ns, roundtrip_ns), strict=True
                ):
                    series[op].append(duration)
    return samples


def _cold_results(
    settings: _CodecSettings,
    raster: Raster,
    expected: Raster,
    packet: bytes,
    repetitions: int,
) -> dict[str, list[dict[str, object]]]:
    """Check the first packet and raster from each separate process."""
    cold: dict[str, list[dict[str, object]]] = {backend: [] for backend in _BACKENDS}
    expected_packet = hashlib.sha256(packet).hexdigest()
    expected_raster = hashlib.sha256(expected.tobytes(order="C")).hexdigest()
    with TemporaryDirectory(prefix="predictive-benchmark-") as directory:
        path = Path(directory) / "raster.npy"
        np.save(path, raster, allow_pickle=False)
        for iteration in range(repetitions):
            order = _BACKENDS if iteration % 2 == 0 else _BACKENDS[::-1]
            for backend in order:
                argv = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "first-call",
                    str(path),
                    backend,
                    settings.predictor,
                    str(settings.precision),
                ]
                start = perf_counter_ns()
                result = json.loads(subprocess.check_output(argv, text=True, timeout=240))
                result["process_wall_ns"] = perf_counter_ns() - start
                if (
                    result["packet_sha256"] != expected_packet
                    or result["raster_sha256"] != expected_raster
                ):
                    raise ValueError(f"fresh {backend} process changed packet or reconstruction")
                cold[backend].append(result)
    return cold


def _first_call(args: argparse.Namespace) -> None:
    """Run one first-call measurement on the selected resident input."""
    raster = np.load(args.input, allow_pickle=False)
    settings = _CodecSettings(args.predictor, args.precision)
    start = perf_counter_ns()
    codec = settings.build(args.backend)
    setup_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    packet, _ = codec.compress(raster)
    encode_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    decoded = codec.decompress(packet, *raster.shape)
    decode_ns = perf_counter_ns() - start
    print(
        json.dumps(
            {
                "setup_ns": setup_ns,
                "encode_ns": encode_ns,
                "decode_ns": decode_ns,
                "roundtrip_ns": encode_ns + decode_ns,
                "peak_rss_bytes": _peak_rss(),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "raster_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
            },
            allow_nan=False,
        )
    )


def _recordings(path: Path | None, selected: list[int]) -> Iterator[tuple[str, Raster]]:
    """Read the selected complete recordings, or the deterministic synthetic cohort."""
    if path is None:
        yield from _synthetic()
        return
    remaining = set(map(str, selected))
    for key, raster in _shd(path):
        if key in remaining:
            remaining.remove(key)
            yield key, raster
            if not remaining:
                return
    raise ValueError(f"SHD recording indices unavailable: {', '.join(sorted(remaining))}")


def _host() -> dict[str, object]:
    """Record the executing CPU, affinity and interpreter settings."""
    cpuinfo = Path("/proc/cpuinfo")
    model = platform.processor()
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpu_root = Path("/sys/devices/system/cpu")
    core_topology = {}
    for cpu in affinity:
        siblings = cpu_root / f"cpu{cpu}" / "topology/thread_siblings_list"
        if siblings.is_file():
            core_topology[str(cpu)] = siblings.read_text().strip()
    return {
        "platform": platform.platform(),
        "processor": model,
        "logical_cpus": os.cpu_count(),
        "affinity": affinity,
        "thread_siblings": core_topology,
        "host_load_averages": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "python": sys.version,
        "numpy": np.__version__,
    }


def _compare(args: argparse.Namespace) -> None:
    """Validate both packet paths before and after the paired timings."""
    started = datetime.now(UTC).isoformat()
    hardware_and_runtime = _host()
    settings = _CodecSettings(args.predictor, args.precision)
    codecs = settings.build("python"), settings.build("rust")
    timings: list[dict[str, object]] = []
    cold: dict[str, object] = {}
    for name, raster in _recordings(args.shd, args.recording):
        expected = _reconstruction(raster, settings.precision)
        packet = _validate(codecs, raster, expected)
        if not timings:
            cold = {
                "recording": name,
                "samples": _cold_results(settings, raster, expected, packet, args.cold_repetitions),
            }
        samples = _warm_samples(codecs, raster, packet, args.warmups, args.repetitions)
        if _validate(codecs, raster, expected) != packet:
            raise ValueError(f"predictive encoding changed after timing recording {name}")
        timings.append(
            {
                "recording": name,
                "shape": raster.shape,
                "source_events": int(np.count_nonzero(raster)),
                "reconstructed_events": int(np.count_nonzero(expected)),
                "source_sha256": hashlib.sha256(raster.tobytes(order="C")).hexdigest(),
                "reconstructed_sha256": hashlib.sha256(expected.tobytes(order="C")).hexdigest(),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "packet_bytes": len(packet),
                "bit_ratio": raster.size / (8 * len(packet)),
                "samples_ns": samples,
                "summary": {
                    backend: {
                        op: summarize_samples(series, raster.size) for op, series in times.items()
                    }
                    for backend, times in samples.items()
                },
                "paired_python_over_rust_median": {
                    op: statistics.median(
                        a / b
                        for a, b in zip(samples["python"][op], samples["rust"][op], strict=True)
                    )
                    for op in OPERATIONS
                },
            }
        )
    modules = [
        "sc_neurocore.spike_codec.predictive_codec",
        "sc_neurocore.spike_codec._predictive_kernels",
        "sc_neurocore.spike_codec.codec",
        "sc_neurocore.spike_codec.isi_packet",
        "sc_neurocore.spike_codec.entropy",
        "sc_neurocore_engine.sc_neurocore_engine",
    ]
    if settings.predictor == "world_model":
        modules.append("sc_neurocore.spike_codec._world_model")
    report = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "hardware_and_runtime": hardware_and_runtime,
        "finished_host_load_averages": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "host_profile": args.host_profile,
        "predictor": settings.predictor,
        "precision": settings.precision,
        "dataset": "synthetic" if args.shd is None else "Spiking Heidelberg Digits",
        "dataset_sha256": None if args.shd is None else _hash(args.shd),
        "shd_recordings": sorted(args.recording),
        "shd_rasterisation": "binary, 700 channels, floor(seconds * 1000)" if args.shd else None,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cold_repetitions": args.cold_repetitions,
        "modules_sha256": {module: _module_hash(module) for module in modules},
        "harness_sha256": {
            name: _hash(Path(__file__).with_name(name))
            for name in (Path(__file__).name, "bench_isi_codec.py", "isi_codec_timing.py")
        },
        "fresh_process": cold,
        "rows": timings,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Run an exact comparison or measure one fresh backend process."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--predictor", choices=_PREDICTORS, required=True)
    compare.add_argument("--precision", type=int, default=1)
    compare.add_argument("--warmups", type=int, default=2)
    compare.add_argument("--repetitions", type=int, default=7)
    compare.add_argument("--cold-repetitions", type=int, default=5)
    compare.add_argument("--shd", type=Path)
    compare.add_argument("--recording", type=int, action="append", default=[])
    compare.add_argument("--host-profile", required=True, choices=("isolated", "shared"))
    compare.add_argument("--output", type=Path, required=True)
    first = commands.add_parser("first-call")
    first.add_argument("input", type=Path)
    first.add_argument("backend", choices=_BACKENDS)
    first.add_argument("predictor", choices=_PREDICTORS)
    first.add_argument("precision", type=int)
    args = parser.parse_args()
    if args.precision < 1:
        parser.error("precision must be positive")
    if args.command == "first-call":
        _first_call(args)
        return
    if args.warmups < 1 or min(args.repetitions, args.cold_repetitions) < 5:
        parser.error("use at least one warmup and five repetitions in each timing group")
    if bool(args.shd) != bool(args.recording) or min(args.recording, default=0) < 0:
        parser.error("select explicit nonnegative recording indices with --shd and --recording")
    if len(set(args.recording)) != len(args.recording):
        parser.error("each SHD recording must be selected once")
    if args.output.exists():
        parser.error("output already exists")
    _compare(args)


if __name__ == "__main__":
    main()

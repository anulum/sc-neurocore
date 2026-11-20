# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Address-event packet and UDP benchmark

"""Compare complete Python and Rust AER packets and measure UDP separately."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

import numpy as np

from bench_isi_codec import Raster, _hash, _module_hash
from bench_predictive_codec import _host, _recordings
from isi_codec_timing import OPERATIONS, _peak_rss, summarize_samples
from sc_neurocore.comm import AEREvent, AERReceiver, AERSender, receive_aer_packet, send_aer_packet
from sc_neurocore.comm.aer_udp import EVENT_SIZE, HEADER_SIZE, MAX_EVENTS_PER_PACKET
from sc_neurocore.spike_codec import AERSpikeCodec

_BACKENDS = ("python", "rust")


def _inputs(args: argparse.Namespace) -> Iterator[tuple[str, Raster]]:
    yield from _recordings(args.shd, args.recording)
    if args.shd is None:
        yield (
            "inverted_dense",
            (np.random.default_rng(185).random((1024, 64)) < 0.8).astype(np.int8),
        )
        gaps = np.zeros((65546, 4), dtype=np.int8)
        gaps[0, 1] = gaps[-1, 3] = 1
        yield "long_timestamp_gap", gaps


def _raster_digest(raster: Raster) -> str:
    return sha256(raster.tobytes(order="C")).hexdigest()


def _sample(codec: AERSpikeCodec, raster: Raster, packet: bytes) -> tuple[int, int, int]:
    start = perf_counter_ns()
    codec.compress(raster)
    encoded = perf_counter_ns()
    codec.decompress(packet)
    decoded = perf_counter_ns()
    codec.decompress(codec.compress(raster)[0])
    return encoded - start, decoded - encoded, perf_counter_ns() - decoded


def _first_call(args: argparse.Namespace) -> None:
    start = perf_counter_ns()
    raster = np.load(args.raster, allow_pickle=False)
    codec = AERSpikeCodec(args.timestamp_bits, args.neuron_bits, backend=args.backend)
    loaded = perf_counter_ns()
    packet, _ = codec.compress(raster)
    encoded = perf_counter_ns()
    decoded = codec.decompress(packet)
    finished = perf_counter_ns()
    print(
        json.dumps(
            {
                "load_and_construct_ns": loaded - start,
                "encode_ns": encoded - loaded,
                "decode_ns": finished - encoded,
                "roundtrip_ns": finished - loaded,
                "packet_sha256": sha256(packet).hexdigest(),
                "decoded_sha256": _raster_digest(decoded),
                "peak_rss_bytes": _peak_rss(),
            }
        )
    )


def _cold(
    raster_file: Path, packet: bytes, raster: Raster, args: argparse.Namespace
) -> dict[str, list[dict[str, object]]]:
    packet_hash, raster_hash = sha256(packet).hexdigest(), _raster_digest(raster)
    results: dict[str, list[dict[str, object]]] = {name: [] for name in _BACKENDS}
    for trial in range(args.cold_repetitions):
        order = _BACKENDS if trial % 2 == 0 else tuple(reversed(_BACKENDS))
        for backend in order:
            command = [
                sys.executable,
                "-B",
                __file__,
                "first-call",
                str(raster_file),
                backend,
                str(args.timestamp_bits),
                str(args.neuron_bits),
            ]
            start = perf_counter_ns()
            actual: dict[str, object] = json.loads(
                subprocess.check_output(command, text=True, timeout=240)
            )
            actual["fresh_process_wall_ns"] = perf_counter_ns() - start
            if actual["packet_sha256"] != packet_hash or actual["decoded_sha256"] != raster_hash:
                raise ValueError("fresh interpreter changed packet or reconstructed raster")
            results[backend].append(actual)
    return results


def _transport(
    codecs: dict[str, AERSpikeCodec], raster: Raster, warmups: int, repetitions: int
) -> dict[str, object]:
    first = int(np.flatnonzero(raster.any(axis=1))[0])
    width = min(32, max(1, (MAX_EVENTS_PER_PACKET * 8) // raster.shape[1]))
    preview = raster[first : first + width]
    positions = np.argwhere(preview)
    raw_events = [AEREvent(int(t), int(n)) for t, n in positions[:MAX_EVENTS_PER_PACKET]]
    expected = codecs["python"].compress(preview)[0]
    raw_samples: list[int] = []
    adapter_samples: dict[str, list[int]] = {name: [] for name in _BACKENDS}
    with (
        closing(AERReceiver(port=0, timeout=2.0)) as receiver,
        closing(AERSender(port=receiver.bound_port)) as sender,
    ):
        for trial in range(warmups + repetitions):
            sequence = sender.next_sequence
            start = perf_counter_ns()
            sender.send(raw_events)
            result = receiver.receive_packet()
            elapsed = perf_counter_ns() - start
            if result is None or result.sequence != sequence or result.events != tuple(raw_events):
                raise ValueError("raw UDP loopback changed events or sequence")
            if trial >= warmups:
                raw_samples.append(elapsed)
            order = _BACKENDS if trial % 2 == 0 else tuple(reversed(_BACKENDS))
            for backend in order:
                codec = codecs[backend]
                start = perf_counter_ns()
                transfer = send_aer_packet(sender, expected, codec=codec)
                restored = receive_aer_packet(receiver, transfer, codec=codec)
                elapsed = perf_counter_ns() - start
                if restored.packet != expected or restored.reordered_datagrams:
                    raise ValueError("UDP raster loopback changed packet or ordering")
                if trial >= warmups:
                    adapter_samples[backend].append(elapsed)
    packet_count = (len(positions) + MAX_EVENTS_PER_PACKET - 1) // MAX_EVENTS_PER_PACKET
    return {
        "raster_row_start": first,
        "raster_shape": list(preview.shape),
        "raster_sha256": _raster_digest(preview),
        "aer_packet_sha256": sha256(expected).hexdigest(),
        "raw_udp_events": len(raw_events),
        "raw_udp_payload_bytes": HEADER_SIZE + len(raw_events) * EVENT_SIZE,
        "raw_udp_samples_ns": raw_samples,
        "adapter_datagrams": packet_count,
        "adapter_events": len(positions),
        "adapter_udp_payload_bytes": packet_count * HEADER_SIZE + len(positions) * EVENT_SIZE,
        "adapter_samples_ns": adapter_samples,
    }


def _row(label: str, raster: Raster, raster_file: Path, args: argparse.Namespace) -> dict[str, Any]:
    codecs = {
        name: AERSpikeCodec(args.timestamp_bits, args.neuron_bits, backend=name)
        for name in _BACKENDS
    }
    encoded = codecs["python"].compress(raster)
    packet = encoded[0]
    samples: dict[str, dict[str, list[int]]] = {
        backend: {op: [] for op in OPERATIONS} for backend in _BACKENDS
    }
    for codec in codecs.values():
        if codec.compress(raster) != encoded or not np.array_equal(
            codec.decompress(packet), raster
        ):
            raise ValueError("packet, statistics or raster parity failed")
    for trial in range(args.warmups + args.repetitions):
        order = _BACKENDS if trial % 2 == 0 else tuple(reversed(_BACKENDS))
        for backend in order:
            durations = _sample(codecs[backend], raster, packet)
            if trial >= args.warmups:
                for op, elapsed in zip(OPERATIONS, durations, strict=True):
                    samples[backend][op].append(elapsed)
    for codec in codecs.values():
        if codec.compress(raster) != encoded or not np.array_equal(
            codec.decompress(packet), raster
        ):
            raise ValueError("packet changed during measurement")
    return {
        "recording": label,
        "shape": list(raster.shape),
        "raster_sha256": _raster_digest(raster),
        "packet_sha256": sha256(packet).hexdigest(),
        "packet_bytes": len(packet),
        "wire_format": packet[:4].decode("ascii"),
        "events": encoded[1].n_events,
        "spikes": encoded[1].n_spikes,
        "samples_ns": samples,
        "summary": {
            name: {op: summarize_samples(series, raster.size) for op, series in timing.items()}
            for name, timing in samples.items()
        },
        "paired_python_over_rust": {
            op: statistics.median(
                py / rust
                for py, rust in zip(samples["python"][op], samples["rust"][op], strict=True)
            )
            for op in OPERATIONS
        },
        "transport": _transport(codecs, raster, args.warmups, args.repetitions),
        "fresh_process": _cold(raster_file, packet, raster, args),
    }


def _compare(args: argparse.Namespace) -> None:
    started = datetime.now(UTC).isoformat()
    hardware = _host()
    rows = []
    with TemporaryDirectory(prefix="address-event-benchmark-") as directory:
        raster_file = Path(directory) / "raster.npy"
        for label, raster in _inputs(args):
            np.save(raster_file, raster, allow_pickle=False)
            rows.append(_row(label, raster, raster_file, args))
    modules = (
        "sc_neurocore.spike_codec.aer_codec",
        "sc_neurocore.spike_codec.aer_packet",
        "sc_neurocore.comm.aer_raster",
        "sc_neurocore.comm.aer_udp",
        "sc_neurocore_engine.sc_neurocore_engine",
    )
    scripts = (
        "bench_address_events.py",
        "bench_isi_codec.py",
        "bench_predictive_codec.py",
        "isi_codec_timing.py",
    )
    report = {
        "schema": 1,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "hardware_before": hardware,
        "hardware_after": _host(),
        "dataset": "synthetic" if args.shd is None else "Spiking Heidelberg Digits",
        "dataset_sha256": None if args.shd is None else _hash(args.shd),
        "recording_indices": sorted(args.recording),
        "timestamp_bits": args.timestamp_bits,
        "neuron_bits": args.neuron_bits,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cold_repetitions": args.cold_repetitions,
        "harness_sha256": {name: _hash(Path(__file__).with_name(name)) for name in scripts},
        "modules_sha256": {name: _module_hash(name) for name in modules},
        "rows": rows,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


def main() -> None:
    """Run the paired measurements or one isolated first use."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    compare = modes.add_parser("compare")
    compare.add_argument("--timestamp-bits", type=int, default=16)
    compare.add_argument("--neuron-bits", type=int, default=0)
    compare.add_argument("--warmups", type=int, default=2)
    compare.add_argument("--repetitions", type=int, default=9)
    compare.add_argument("--cold-repetitions", type=int, default=5)
    compare.add_argument("--shd", type=Path)
    compare.add_argument("--recording", type=int, action="append", default=[])
    compare.add_argument("--output", type=Path, required=True)
    first = modes.add_parser("first-call")
    first.add_argument("raster", type=Path)
    first.add_argument("backend", choices=_BACKENDS)
    first.add_argument("timestamp_bits", type=int)
    first.add_argument("neuron_bits", type=int)
    args = parser.parse_args()
    if args.mode == "first-call":
        _first_call(args)
        return
    if bool(args.shd) != bool(args.recording) or min(args.recording, default=0) < 0:
        parser.error("use --shd with explicit nonnegative --recording indices")
    if len(set(args.recording)) != len(args.recording):
        parser.error("select each recording once")
    if args.warmups < 1 or min(args.repetitions, args.cold_repetitions) < 5:
        parser.error("use at least one warmup and five repetitions per timing group")
    if args.output.exists():
        parser.error("output already exists")
    _compare(args)


if __name__ == "__main__":
    main()

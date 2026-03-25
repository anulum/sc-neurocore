#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike codec library benchmark

"""Benchmark all five spike codecs on synthetic data for four target systems.

Generates:
    1. BCI implant data: 1024ch, 20kHz, 0.5% firing, sparse + bursty
    2. Neural probe data: 384ch, 30kHz, 3% firing, spatially correlated
    3. Neuromorphic data: 256ch, event-driven, 1% firing
    4. Real-time streaming data: 128ch, 20kHz, 2% firing

Reports compression ratio, encode/decode latency, and per-system
codec recommendation accuracy.
"""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.spike_codec import (
    SpikeCodec,
    PredictiveSpikeCodec,
    DeltaSpikeCodec,
    StreamingSpikeCodec,
    AERSpikeCodec,
    recommend_codec,
)


def generate_bci_data(rng: np.random.RandomState) -> np.ndarray:
    """1024 channels, 1 second at 20kHz, sparse with periodic bursts."""
    T, N = 20000, 1024
    spikes = np.zeros((T, N), dtype=np.int8)
    for n in range(N):
        rate = rng.uniform(0.5, 5.0)  # Hz
        n_spikes = int(rate * T / 20000)
        times = rng.choice(T, size=min(n_spikes, T), replace=False)
        spikes[times, n] = 1
    return spikes


def generate_probe_data(rng: np.random.RandomState) -> np.ndarray:
    """384 channels, correlated groups (Neuropixels-like)."""
    T, N = 10000, 384
    group_size = 8
    spikes = np.zeros((T, N), dtype=np.int8)
    for g in range(N // group_size):
        base = (rng.random(T) < 0.02).astype(np.int8)
        for c in range(group_size):
            ch = g * group_size + c
            noise = (rng.random(T) < 0.005).astype(np.int8)
            spikes[:, ch] = (base | noise).astype(np.int8)
    return spikes


def generate_neuromorphic_data(rng: np.random.RandomState) -> np.ndarray:
    """256 channels, 1% sparse event-driven."""
    T, N = 5000, 256
    return (rng.random((T, N)) < 0.01).astype(np.int8)


def generate_realtime_data(rng: np.random.RandomState) -> np.ndarray:
    """128 channels, 2% firing for closed-loop BCI."""
    T, N = 2000, 128
    return (rng.random((T, N)) < 0.02).astype(np.int8)


def benchmark_codec(codec, spikes: np.ndarray, name: str) -> dict:
    T, N = spikes.shape

    t0 = time.perf_counter()
    data, result = codec.compress(spikes)
    compress_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if hasattr(result, "window_size") or hasattr(result, "n_events"):
        recovered = codec.decompress(data)
    else:
        recovered = codec.decompress(data, T, N)
    decompress_ms = (time.perf_counter() - t0) * 1000

    lossless = bool(np.array_equal(recovered, spikes))

    return {
        "codec": name,
        "ratio": result.compression_ratio,
        "compress_ms": compress_ms,
        "decompress_ms": decompress_ms,
        "compressed_bytes": len(data),
        "lossless": lossless,
    }


def print_table(results: list[dict], scenario: str):
    print(f"\n{'=' * 70}")
    print(f"  {scenario}")
    print(f"{'=' * 70}")
    print(
        f"  {'Codec':<14} {'Ratio':>8} {'Enc (ms)':>10} {'Dec (ms)':>10} {'Bytes':>10} {'Lossless':>9}"
    )
    print(f"  {'-' * 14} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 9}")
    for r in results:
        print(
            f"  {r['codec']:<14} {r['ratio']:>8.1f}x {r['compress_ms']:>9.1f} "
            f"{r['decompress_ms']:>9.1f} {r['compressed_bytes']:>10,} {'yes' if r['lossless'] else 'NO':>9}"
        )


def main():
    rng = np.random.RandomState(42)

    scenarios = [
        ("BCI Implant (1024ch, 20kHz, sparse+bursts)", generate_bci_data(rng)),
        ("Neural Probe (384ch, correlated groups)", generate_probe_data(rng)),
        ("Neuromorphic (256ch, 1% sparse)", generate_neuromorphic_data(rng)),
        ("Real-time Streaming (128ch, 2%)", generate_realtime_data(rng)),
    ]

    codecs = [
        ("isi", SpikeCodec()),
        ("predictive", PredictiveSpikeCodec(alpha=0.005, threshold=0.5)),
        ("delta", DeltaSpikeCodec(group_size=8)),
        ("streaming", StreamingSpikeCodec(window_size=20)),
        ("aer", AERSpikeCodec()),
    ]

    all_results = {}

    for scenario_name, spikes in scenarios:
        T, N = spikes.shape
        n_spikes = int(np.sum(spikes))
        firing_pct = n_spikes / (T * N) * 100
        print(f"\nData: {T}x{N} = {T * N:,} bits, {n_spikes:,} spikes ({firing_pct:.2f}%)")

        results = []
        for codec_name, codec in codecs:
            r = benchmark_codec(codec, spikes, codec_name)
            results.append(r)

        results.sort(key=lambda x: -x["ratio"])
        print_table(results, scenario_name)

        winner = results[0]["codec"]
        recommended = recommend_codec(
            N,
            n_spikes / (N * T / 20000),
            latency_ms=5.0,
            correlated="correlated" in scenario_name.lower(),
            neuromorphic="neuromorphic" in scenario_name.lower(),
        )
        print(f"\n  Best actual: {winner} ({results[0]['ratio']:.1f}x)")
        print(f"  Recommended: {recommended}")

        all_results[scenario_name] = results

    # Verify all codecs are lossless
    all_lossless = all(r["lossless"] for results in all_results.values() for r in results)
    print(f"\n{'=' * 70}")
    print(f"  All codecs lossless: {'YES' if all_lossless else 'FAILURES DETECTED'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

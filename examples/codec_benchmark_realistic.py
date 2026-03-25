#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Realistic spike codec benchmark via SpikeInterface

"""Benchmark all 5 spike codecs on SpikeInterface-generated recordings.

SpikeInterface generates physiologically realistic spike trains with
proper ISI distributions, refractory periods, and firing rate variability.
This is closer to real electrode data than uniform random.

Scenarios:
    1. Neuropixels-like: 96ch, 30kHz, 10 units, 1-5 Hz
    2. BCI-scale: 256ch, 30kHz, 50 units, 0.5-3 Hz
    3. High-density: 384ch, 30kHz, 100 units, 1-10 Hz
"""

from __future__ import annotations

import time

import numpy as np

try:
    import spikeinterface.generation as gen
except ImportError:
    gen = None

from sc_neurocore.spike_codec import (
    SpikeCodec,
    PredictiveSpikeCodec,
    DeltaSpikeCodec,
    StreamingSpikeCodec,
    AERSpikeCodec,
)


def generate_realistic_raster(
    n_channels: int,
    n_units: int,
    duration_s: float,
    fs: float = 30000.0,
    firing_range: tuple = (1.0, 5.0),
    seed: int = 42,
) -> np.ndarray:
    """Generate realistic spike raster via SpikeInterface."""
    if gen is None:
        raise ImportError("spikeinterface required: pip install spikeinterface")

    _, sorting = gen.generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=fs,
        num_channels=min(n_channels, 32),
        num_units=n_units,
        generate_sorting_kwargs=dict(firing_rates=firing_range),
        seed=seed,
    )

    T = int(duration_s * fs)
    N = sorting.get_num_units()
    raster = np.zeros((T, N), dtype=np.int8)
    for i, uid in enumerate(sorting.unit_ids):
        times = sorting.get_unit_spike_train(uid)
        valid = times[times < T]
        raster[valid, i] = 1

    return raster


def benchmark_codec(codec, spikes, name):
    T, N = spikes.shape
    t0 = time.perf_counter()
    data, result = codec.compress(spikes)
    enc_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if hasattr(result, "window_size") or hasattr(result, "n_events"):
        recovered = codec.decompress(data)
    else:
        recovered = codec.decompress(data, T, N)
    dec_ms = (time.perf_counter() - t0) * 1000

    lossless = bool(np.array_equal(recovered, spikes))
    return {
        "codec": name,
        "ratio": result.compression_ratio,
        "enc_ms": enc_ms,
        "dec_ms": dec_ms,
        "bytes": len(data),
        "lossless": lossless,
    }


def print_table(results, scenario):
    print(f"\n{'=' * 72}")
    print(f"  {scenario}")
    print(f"{'=' * 72}")
    print(f"  {'Codec':<20} {'Ratio':>8} {'Enc(ms)':>9} {'Dec(ms)':>9} {'Bytes':>10} {'OK':>5}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 10} {'-' * 5}")
    for r in results:
        ok = "yes" if r["lossless"] else "NO"
        print(
            f"  {r['codec']:<20} {r['ratio']:>8.1f}x {r['enc_ms']:>8.1f} "
            f"{r['dec_ms']:>8.1f} {r['bytes']:>10,} {ok:>5}"
        )


def main():
    codecs = [
        ("isi", SpikeCodec()),
        ("predictive-ema", PredictiveSpikeCodec(alpha=0.005)),
        ("predictive-lfsr", PredictiveSpikeCodec(predictor="lfsr", alpha_q8=1)),
        ("delta-g8", DeltaSpikeCodec(group_size=8)),
        ("streaming-20", StreamingSpikeCodec(window_size=20)),
        ("aer", AERSpikeCodec()),
    ]

    scenarios = [
        ("Neuropixels-like (10 units, 1-5 Hz, 2s)", 96, 10, 2.0, (1.0, 5.0)),
        ("BCI-scale (50 units, 0.5-3 Hz, 2s)", 256, 50, 2.0, (0.5, 3.0)),
        ("High-density (100 units, 1-10 Hz, 1s)", 384, 100, 1.0, (1.0, 10.0)),
    ]

    all_lossless = True

    for name, nch, nunits, dur, frates in scenarios:
        print(f"\nGenerating: {name}...")
        raster = generate_realistic_raster(nch, nunits, dur, firing_range=frates)
        T, N = raster.shape
        n_spikes = int(raster.sum())
        sparsity = n_spikes / (T * N) * 100
        print(f"  Raster: ({T:,} x {N}), {n_spikes:,} spikes ({sparsity:.4f}%)")

        results = []
        for cname, codec in codecs:
            r = benchmark_codec(codec, raster, cname)
            results.append(r)
            if not r["lossless"]:
                all_lossless = False

        results.sort(key=lambda x: -x["ratio"])
        print_table(results, name)

    print(f"\n{'=' * 72}")
    print(f"  All codecs lossless: {'YES' if all_lossless else 'FAILURES DETECTED'}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

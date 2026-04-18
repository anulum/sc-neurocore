#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WaveformCodec demo: raw electrode -> compressed telemetry

r"""Neural implant waveform compression demo.

Demonstrates the full Neuralink-scale compression pipeline:
    Raw 10-bit ADC (1024ch, 20kHz) -> spike detect -> compress -> telemetry

This is the pipeline that fits 328 Mbit/s raw data into a 10-20 Mbit/s
wireless uplink. Spike timing is lossless. Waveform shapes are preserved
via template matching. Background LFP is lossy-compressed.

Usage:
    python examples/demo_waveform_codec.py
    python examples/demo_waveform_codec.py --channels 1024 --duration 1.0
    python examples/demo_waveform_codec.py --channels 384 --quantize 6  # Neuropixels
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def generate_realistic_waveform(
    n_channels: int,
    duration_s: float,
    fs: float = 20000.0,
    noise_uv: float = 50.0,
    firing_rate_range: tuple = (0.5, 5.0),
    seed: int = 42,
) -> np.ndarray:
    """Generate realistic multi-channel electrode recording.

    Includes:
    - Gaussian background noise (models thermal + biological noise)
    - Biphasic spike waveforms injected at random times
    - Per-channel variable firing rates (0.5-5 Hz, cortical range)
    - 3 distinct spike templates (modeling different neuron types)
    """
    rng = np.random.RandomState(seed)
    T = int(duration_s * fs)

    # Background noise
    waveform = rng.randn(T, n_channels).astype(np.float32) * noise_uv

    # 3 spike templates (different neuron morphologies)
    templates = []
    # Type 1: Fast-spiking interneuron (narrow spike)
    t1 = np.zeros(48, dtype=np.float32)
    t1[18:22] = -250
    t1[22:26] = 150
    t1[26:28] = -40
    templates.append(t1)

    # Type 2: Regular-spiking pyramidal (broader spike)
    t2 = np.zeros(48, dtype=np.float32)
    t2[16:22] = -180
    t2[22:28] = 120
    t2[28:32] = -50
    t2[32:36] = 20
    templates.append(t2)

    # Type 3: Bursting neuron (triphasic)
    t3 = np.zeros(48, dtype=np.float32)
    t3[14:18] = 30
    t3[18:22] = -220
    t3[22:26] = 160
    t3[26:30] = -80
    t3[30:34] = 30
    templates.append(t3)

    total_spikes = 0
    for ch in range(n_channels):
        rate = rng.uniform(*firing_rate_range)
        n_spikes = max(1, int(rate * duration_s))
        tmpl_idx = ch % len(templates)
        tmpl = templates[tmpl_idx] * rng.uniform(0.7, 1.3)

        times = rng.choice(range(100, T - 100), size=min(n_spikes, T - 200), replace=False)
        for t in times:
            s, e = max(0, t - 24), min(T, t + 24)
            waveform[s:e, ch] += tmpl[: e - s]
        total_spikes += len(times)

    return waveform, total_spikes


def main():
    parser = argparse.ArgumentParser(description="SC-NeuroCore WaveformCodec Demo")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in seconds")
    parser.add_argument("--fs", type=float, default=20000.0, help="Sampling rate (Hz)")
    parser.add_argument("--quantize", type=int, default=4, help="Background quantization bits")
    parser.add_argument(
        "--threshold", type=float, default=4.5, help="Spike detection threshold (sigma)"
    )
    args = parser.parse_args()

    from sc_neurocore.spike_codec import WaveformCodec

    print("=" * 70)
    print("  SC-NeuroCore WaveformCodec — Neural Implant Compression Demo")
    print("=" * 70)

    # Generate data
    print(
        f"\nGenerating {args.channels}-channel recording ({args.duration}s at {args.fs / 1000:.0f} kHz)..."
    )
    waveform, injected_spikes = generate_realistic_waveform(
        n_channels=args.channels,
        duration_s=args.duration,
        fs=args.fs,
    )
    T, N = waveform.shape
    raw_bytes = T * N * 2
    raw_mbits = raw_bytes * 8 / 1e6
    bandwidth_mbps = raw_mbits / args.duration

    print(f"  Shape: ({T:,} samples x {N} channels)")
    print(f"  Raw size: {raw_bytes:,} bytes ({raw_mbits:.1f} Mbit)")
    print(f"  Raw bandwidth: {bandwidth_mbps:.0f} Mbit/s")
    print(f"  Injected spikes: {injected_spikes:,}")

    # Compress
    print(f"\nCompressing (threshold={args.threshold}sigma, background={args.quantize}-bit)...")
    codec = WaveformCodec(
        threshold_sigma=args.threshold,
        snippet_samples=48,
        max_templates=16,
        quantize_bits=args.quantize,
    )

    t0 = time.perf_counter()
    data, result = codec.compress(waveform)
    enc_time = time.perf_counter() - t0

    compressed_mbits = result.compressed_bytes * 8 / 1e6
    compressed_bw = compressed_mbits / args.duration

    print("\n  Compression Results:")
    print(f"  {'-' * 50}")
    print(f"  Raw:        {raw_bytes:>12,} bytes ({bandwidth_mbps:.0f} Mbit/s)")
    print(f"  Compressed: {result.compressed_bytes:>12,} bytes ({compressed_bw:.1f} Mbit/s)")
    print(f"  Ratio:      {result.compression_ratio:>12.1f}x")
    print(f"  {'-' * 50}")
    print(f"  Spikes detected:    {result.n_spikes_detected:,}")
    print(f"  Templates learned:  {result.n_templates}")
    print(
        f"  Spike timing:       {result.spike_bytes:>10,} bytes ({result.spike_bytes / result.compressed_bytes * 100:.1f}%)"
    )
    print(
        f"  Waveform snippets:  {result.snippet_bytes:>10,} bytes ({result.snippet_bytes / result.compressed_bytes * 100:.1f}%)"
    )
    print(
        f"  Background LFP:     {result.background_bytes:>10,} bytes ({result.background_bytes / result.compressed_bytes * 100:.1f}%)"
    )
    print(f"  Encode time:        {enc_time * 1000:.0f} ms")
    print(f"  Spike timing:       {'LOSSLESS' if result.lossless_spikes else 'LOSSY'}")

    # Neuralink comparison
    bt_capacity_mbps = 15.0  # typical Bluetooth uplink
    fits = compressed_bw < bt_capacity_mbps
    print("\n  Neuralink Uplink Check:")
    print(f"  {'-' * 50}")
    print(f"  Bluetooth capacity: {bt_capacity_mbps:.0f} Mbit/s")
    print(f"  Compressed rate:    {compressed_bw:.1f} Mbit/s")
    print(f"  Fits in uplink:     {'YES' if fits else 'NO — need more compression'}")

    if fits:
        headroom = (1 - compressed_bw / bt_capacity_mbps) * 100
        print(f"  Headroom:           {headroom:.0f}%")

    # Scaling table
    print("\n  Scaling to different channel counts:")
    print(f"  {'Channels':>10} {'Raw Mbit/s':>12} {'Compressed':>12} {'Ratio':>8} {'Fits BT':>8}")
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 8}")

    for nch in [128, 256, 384, 512, 1024, 3072]:
        raw_bw = nch * args.fs * 10 / 1e6  # 10-bit
        est_compressed = raw_bw / result.compression_ratio
        fits_bt = est_compressed < bt_capacity_mbps
        print(
            f"  {nch:>10} {raw_bw:>11.0f} {est_compressed:>11.1f} "
            f"{result.compression_ratio:>7.1f}x {'YES' if fits_bt else 'NO':>7}"
        )

    print(f"\n{'=' * 70}")
    print("  Pipeline: Raw ADC -> Spike Detect -> Template Match -> Compress")
    print("  Spike timing: lossless (ISI + Huffman entropy)")
    print("  Waveform shapes: template library + quantized residuals")
    print("  Background: delta + quantize + zlib")
    print("  Hardware: bit-true LFSR predictor maps to Verilog RTL")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

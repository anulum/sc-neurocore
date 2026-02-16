#!/usr/bin/env python3
"""
SSGF Adaptive Audio Demo (Standalone Terminal)
===============================================

Simulates a full adaptive audio session:
1. Baseline EEG recording (simulated)
2. Session with EVS feedback driving SSGF adaptation
3. Shows phase transitions: Discovery → Lock-On → Deepening

Usage:
    python -m sc_neurocore.experiments.demo_adaptive_audio

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import numpy as np

from ..audio import (
    SSGFEngine, SSGFConfig,
    EVSEngine, EVSConfig,
    AdaptiveAudioEngine, AdaptiveConfig,
    UserProfile, Chronotype,
)


def generate_eeg_signal(
    target_hz: float, t: float, sample_rate: int, entrained: float = 0.5
) -> float:
    """Generate a simulated EEG sample with variable entrainment strength."""
    # Base signal: target frequency with variable amplitude
    signal = entrained * np.sin(2 * np.pi * target_hz * t / sample_rate)
    # Background: pink noise approximation
    noise = np.random.normal(0, 0.3 * (1.0 - entrained * 0.5))
    # Alpha background
    alpha = 0.2 * np.sin(2 * np.pi * 10.0 * t / sample_rate)
    return float(signal + noise + alpha)


def run_demo():
    """Run full adaptive audio demo."""
    print("=" * 72)
    print("  SSGF Adaptive Audio Personalization Demo")
    print("  SC-NeuroCore — Real-Time Entrainment Adaptation")
    print("=" * 72)

    # Setup
    profile = UserProfile(user_id="demo", chronotype=Chronotype.BEAR)
    target_hz = profile.get_optimal_target_hz()
    print(f"\n  User: {profile.user_id} | Chronotype: {profile.chronotype.value}")
    print(f"  Optimal target: {target_hz} Hz")

    ssgf = SSGFEngine(SSGFConfig(N=8, micro_steps=5, seed=42))
    evs = EVSEngine(EVSConfig(target_hz=target_hz, sample_rate=256, fft_window=256))
    adaptive = AdaptiveAudioEngine(
        ssgf, evs, profile,
        AdaptiveConfig(
            phase1_duration_s=30,   # Shortened for demo
            phase2_duration_s=70,
        ),
    )

    # Phase 1: Baseline
    print("\n--- Baseline Recording (simulated) ---")
    evs.start_baseline()
    for t in range(512):
        sample = generate_eeg_signal(target_hz, t, 256, entrained=0.0)
        evs.add_sample(sample)
    baseline = evs.stop_baseline()
    print(f"  Baseline powers: {', '.join(f'{k}={v:.2f}' for k, v in baseline.items())}")

    # Phase 2: Adaptive session
    print("\n--- Adaptive Session ---")
    evs.start_session(target_hz)
    adaptive.start_session()

    print(f"  {'Tick':>5} | {'Phase':>12} | {'EVS':>5} | {'Trend':>6} | {'R':>5} | {'Adaptations'}")
    print(f"  {'─' * 5} | {'─' * 12} | {'─' * 5} | {'─' * 6} | {'─' * 5} | {'─' * 30}")

    for tick in range(150):
        # Simulate improving entrainment over time
        entrainment_strength = min(0.9, 0.1 + tick * 0.005)

        # Add EEG samples (1 EVS compute per ~10 samples)
        for s in range(10):
            t_global = tick * 10 + s + 512
            sample = generate_eeg_signal(target_hz, t_global, 256, entrained=entrainment_strength)
            evs.add_sample(sample)

        # Compute EVS
        evs_snap = evs.compute()

        # Adaptive feedback
        adapt_snap = adaptive.on_evs_update(evs_snap)

        if tick % 15 == 0 or tick == 149:
            adaptations = ", ".join(adapt_snap.adaptations_applied) or "none"
            print(
                f"  {tick:5d} | {adapt_snap.session_phase:>12} | "
                f"{adapt_snap.evs_score:5.1f} | {adapt_snap.evs_trend:+6.2f} | "
                f"{adapt_snap.R_global:5.3f} | {adaptations}"
            )

    adaptive.stop_session()

    # Phase 3: Report
    print("\n--- Session Report ---")
    report = adaptive.get_session_report()
    print(f"  Duration: {report.total_ticks} ticks")
    print(f"  EVS Average: {report.evs_avg:.1f}")
    print(f"  EVS Peak: {report.evs_peak:.1f}")
    print(f"  Verified Time: {report.time_verified_pct:.1f}%")
    print(f"  Theurgic Time: {report.theurgic_time_pct:.1f}%")
    print(f"  Grade: {report.grade}")
    print(f"  Phase breakdown: {report.phase_durations}")

    print(f"\n  Updated profile:")
    print(f"    Sessions: {profile.session_count}")
    print(f"    Best EVS: {profile.best_evs_score:.1f}")
    print(f"    Sensitivity: {profile.sensitivity_map}")

    print(f"\n{'=' * 72}")
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()

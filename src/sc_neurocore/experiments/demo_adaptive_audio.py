#!/usr/bin/env python3
"""
Adaptive Audio Demo -- SSGF + EVS Closed-Loop
===============================================

Creates a UserProfile (BEAR chronotype), SSGFEngine, EVSEngine, and
AdaptiveAudioEngine, then runs 50 ticks with simulated EEG showing
real-time EVS scores and audio parameter adaptation.

Usage:
    python -m sc_neurocore.experiments.demo_adaptive_audio

Author: Claude (Session 2026-02-17)
"""

from __future__ import annotations

import numpy as np

from ..audio.ssgf_engine import SSGFConfig, SSGFEngine
from ..audio.evs_engine import EVSConfig, EVSEngine
from ..audio.adaptive_engine import AdaptiveAudioEngine
from ..audio.user_profile import UserProfile, Chronotype


def _generate_eeg_chunk(
    rng: np.random.RandomState,
    n_samples: int,
    target_hz: float,
    sample_rate: int,
    entrained_strength: float = 0.5,
    noise_level: float = 0.3,
) -> np.ndarray:
    """Generate simulated EEG with a target-frequency component + noise.

    Parameters
    ----------
    rng : RandomState
        Random number generator.
    n_samples : int
        Number of samples to produce.
    target_hz : float
        Entrainment target frequency.
    sample_rate : int
        Sampling rate in Hz.
    entrained_strength : float
        Amplitude of the entrained sinusoidal component (0-1).
    noise_level : float
        Amplitude of background noise.

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    t = np.arange(n_samples) / sample_rate
    # Target sinusoid
    signal = entrained_strength * np.sin(2 * np.pi * target_hz * t)
    # Background: mix of brain bands
    signal += 0.15 * np.sin(2 * np.pi * 2.5 * t)  # delta
    signal += 0.10 * np.sin(2 * np.pi * 6.0 * t)  # theta
    signal += 0.08 * np.sin(2 * np.pi * 20.0 * t)  # beta
    # Noise
    signal += noise_level * rng.randn(n_samples)
    return signal


def run_demo():
    rng = np.random.RandomState(42)

    # ── 1. Create components ─────────────────────────────────────────

    profile = UserProfile(
        user_id="demo_user",
        chronotype=Chronotype.BEAR,
    )

    ssgf_cfg = SSGFConfig(
        N=16,
        z_dim=120,
        lr_z=0.01,
        sigma_g=0.3,
        micro_steps=10,
        dt=0.001,
        noise=0.2,
        K_base=0.45,
        K_alpha=0.3,
        field_pressure=0.1,
        seed=42,
    )
    ssgf = SSGFEngine(ssgf_cfg)

    evs_cfg = EVSConfig(
        sample_rate=256,
        fft_window=512,
        baseline_duration_s=2.0,  # short for demo
        update_interval_samples=128,
    )
    evs = EVSEngine(evs_cfg)

    adaptive = AdaptiveAudioEngine(ssgf, evs, profile)

    target_hz = profile.get_best_target_hz()

    # ── 2. Baseline ──────────────────────────────────────────────────

    print("=" * 72)
    print("  SSGF Adaptive Audio Demo")
    print("  Chronotype: BEAR | Target: %.1f Hz (alpha)" % target_hz)
    print("=" * 72)
    print("\n  Recording baseline...")

    evs.start_baseline()
    baseline_samples = int(evs_cfg.baseline_duration_s * evs_cfg.sample_rate)
    baseline_eeg = _generate_eeg_chunk(
        rng,
        baseline_samples,
        target_hz,
        evs_cfg.sample_rate,
        entrained_strength=0.1,  # low during baseline
        noise_level=0.5,
    )
    for v in baseline_eeg:
        evs.add_sample(float(v))

    print("  Baseline done: %s" % ({k: "%.4f" % v for k, v in evs._baseline_powers.items()}))

    evs.set_target(target_hz)

    # ── 3. Run 50 adaptive ticks ─────────────────────────────────────

    print(
        "\n  %s  %s  %s  %s  %s  %s  %s"
        % (
            "Tick".rjust(6),
            "Phase".ljust(10),
            "EVS".rjust(6),
            "Verif".rjust(6),
            "Bin.Hz".rjust(7),
            "R".rjust(7),
            "Theurgic".rjust(8),
        )
    )
    print("  " + "-" * 62)

    samples_per_tick = evs_cfg.update_interval_samples

    for tick in range(1, 51):
        # Simulate increasing entrainment over time
        progress = tick / 50.0
        entrained_strength = 0.2 + 0.6 * progress
        noise_level = 0.5 - 0.3 * progress

        chunk = _generate_eeg_chunk(
            rng,
            samples_per_tick,
            target_hz,
            evs_cfg.sample_rate,
            entrained_strength=entrained_strength,
            noise_level=noise_level,
        )

        for v in chunk:
            evs.add_sample(float(v))

        snapshot = evs.compute()
        if snapshot is None:
            continue

        audio = adaptive.on_evs_update(snapshot)

        verified_str = " YES" if snapshot.is_verified else "  no"
        theurgic_str = " YES" if audio.get("theurgic_mode", False) else "  no"

        print(
            "  %s  %s  %s  %s  %s  %s  %s"
            % (
                str(tick).rjust(6),
                adaptive.current_phase.value.ljust(10),
                ("%.1f" % snapshot.evs_score).rjust(6),
                verified_str.rjust(6),
                ("%.2f" % audio["binaural_hz"]).rjust(7),
                ("%.4f" % audio["intensity"]).rjust(7),
                theurgic_str.rjust(8),
            )
        )

    # ── 4. Session Report ────────────────────────────────────────────

    report = adaptive.get_session_report()

    print("\n" + "=" * 72)
    print("  Session Report")
    print("=" * 72)
    print("  Total ticks:    %d" % report.total_ticks)
    print("  Average EVS:    %.2f" % report.avg_evs)
    print("  Peak EVS:       %.2f" % report.peak_evs)
    print("  Verified %%:     %.1f%%" % report.verified_pct)
    print("  Grade:          %s" % report.grade)
    print("  Adaptations:    %d" % report.adaptations)
    print("  Phase durations: %s" % report.phase_durations)
    print(
        "  Final audio:    %s"
        % {
            k: ("%.3f" % v if isinstance(v, float) else str(v))
            for k, v in report.final_audio.items()
        }
    )

    # ── 5. Update profile ────────────────────────────────────────────

    profile.update_from_session(
        avg_evs=report.avg_evs,
        peak_evs=report.peak_evs,
        best_target_hz=target_hz,
        band_powers=evs._baseline_powers,
    )

    print("\n  Profile after session:")
    print("    Sessions:       %d" % profile.session_count)
    print("    Target Hz:      %.2f" % profile.get_best_target_hz())
    print("    Chronotype:     %s" % profile.chronotype.value)

    print("\n" + "=" * 72)
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()

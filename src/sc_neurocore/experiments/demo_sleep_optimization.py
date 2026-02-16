#!/usr/bin/env python3
"""
Sleep Optimization Demo (Standalone Terminal)
==============================================

Simulates an accelerated overnight sleep session:
1. Circadian profiling
2. Protocol selection
3. Closed-loop sleep with simulated EEG
4. Morning report

Usage:
    python -m sc_neurocore.experiments.demo_sleep_optimization

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import numpy as np

from ..sleep import (
    SleepOptimizer, SleepOptimizerConfig,
    CircadianOptimizer, Chronotype,
    SleepReportGenerator,
    SleepStage,
    list_protocols, get_protocol,
)


def generate_sleep_eeg(
    stage: SleepStage, sample_rate: int = 256, n_samples: int = 256
) -> np.ndarray:
    """Generate simulated EEG for a given sleep stage."""
    t = np.arange(n_samples) / sample_rate
    rng = np.random.RandomState()

    # Stage-specific dominant frequencies
    if stage == SleepStage.WAKE:
        signal = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    elif stage == SleepStage.N1:
        signal = 0.6 * np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 10 * t)
    elif stage == SleepStage.N2:
        signal = 0.5 * np.sin(2 * np.pi * 4 * t) + 0.4 * np.sin(2 * np.pi * 2 * t)
    elif stage == SleepStage.N3:
        signal = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t)
    elif stage == SleepStage.REM:
        signal = 0.4 * np.sin(2 * np.pi * 6 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    else:
        signal = np.zeros(n_samples)

    noise = rng.normal(0, 0.2, n_samples)
    return signal + noise


def simulated_sleep_progression(tick: int, total_ticks: int) -> SleepStage:
    """Return expected stage for a realistic overnight progression."""
    progress = tick / max(total_ticks, 1)
    if progress < 0.05:
        return SleepStage.WAKE
    elif progress < 0.10:
        return SleepStage.N1
    elif progress < 0.20:
        return SleepStage.N2
    elif progress < 0.35:
        return SleepStage.N3
    elif progress < 0.45:
        return SleepStage.N2
    elif progress < 0.55:
        return SleepStage.REM
    elif progress < 0.60:
        return SleepStage.N2
    elif progress < 0.70:
        return SleepStage.N3
    elif progress < 0.80:
        return SleepStage.N2
    elif progress < 0.90:
        return SleepStage.REM
    else:
        return SleepStage.N1


def run_demo():
    """Run accelerated overnight sleep demo."""
    print("=" * 72)
    print("  Sleep Optimization System Demo")
    print("  SC-NeuroCore — Closed-Loop Sleep Protocol Engine")
    print("=" * 72)

    # Setup
    chronotype = Chronotype.BEAR
    circadian = CircadianOptimizer(chronotype)
    profile = circadian.to_dict()

    print(f"\n  Chronotype: {chronotype.value}")
    print(f"  Optimal bedtime: {profile['optimal_bedtime_h']:.1f}h")
    print(f"  Optimal wake: {profile['optimal_wake_h']:.1f}h")
    print(f"  Recommended protocol: {profile['recommended_protocol']}")

    print("\n  Available protocols:")
    for p in list_protocols():
        print(f"    - {p['name']}: {p['description'][:60]}")

    # Start session
    protocol_name = "insomnia_relief"
    optimizer = SleepOptimizer(
        chronotype=chronotype,
        config=SleepOptimizerConfig(
            sample_rate=256,
            fft_window=256,
            stage_check_interval=256,
        ),
    )
    optimizer.start_session(protocol_name)
    protocol = get_protocol(protocol_name)

    total_ticks = 100  # Accelerated

    print(f"\n--- Running Session: {protocol_name} ({total_ticks} epochs) ---")
    print(f"  {'Epoch':>5} | {'Time':>7} | {'Detected':>8} | {'Target':>8} | {'Match':>5} | {'Binaural':>8}")
    print(f"  {'─' * 5} | {'─' * 7} | {'─' * 8} | {'─' * 8} | {'─' * 5} | {'─' * 8}")

    for tick_idx in range(total_ticks):
        # Generate EEG for the "true" sleep stage at this point
        true_stage = simulated_sleep_progression(tick_idx, total_ticks)
        eeg = generate_sleep_eeg(true_stage)

        # Feed samples
        optimizer.add_samples(eeg)

        # Check and adapt
        result = optimizer.check_and_adapt()

        if result and tick_idx % 10 == 0:
            match_str = "  YES" if result.stage_match else "   NO"
            print(
                f"  {result.tick:5d} | {result.elapsed_min:6.1f}m | "
                f"{result.current_stage:>8} | {result.target_stage:>8} | "
                f"{match_str} | {result.audio_params.get('binaural_hz', 0):7.1f}Hz"
            )

    optimizer.stop_session()

    # Generate report
    print("\n--- Morning Report ---")
    report_gen = SleepReportGenerator()
    report = report_gen.generate(optimizer)

    print(f"  Total Duration: {report.total_duration_min:.1f} min")
    print(f"  Sleep Onset Latency: {report.sleep_onset_latency_min:.1f} min")
    print(f"  Sleep Efficiency: {report.sleep_efficiency_pct:.1f}%")
    print(f"  Quality Score: {report.quality_score:.1f}/100")
    print(f"  Grade: {report.grade}")
    print(f"  Wakeups: {report.wakeups}")

    print(f"\n  Stage Breakdown:")
    for stage, pct in report.stage_percentages.items():
        target = report.stage_targets.get(stage, 0)
        bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
        print(f"    {stage:>5}: {pct:5.1f}% (target: {target:5.1f}%) |{bar}|")

    print(f"\n  Recommendations:")
    for rec in report.recommendations:
        print(f"    - {rec}")

    print(f"\n{'=' * 72}")
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()

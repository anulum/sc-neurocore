#!/usr/bin/env python3
"""
TCBO Consciousness Detection Demo (Standalone Terminal)
=======================================================

Runs all 5 TCBO scenarios and prints per-step output showing:
- p_h1 consciousness observable
- Gate open/close transitions
- PI controller recovery after perturbation
- Kuramoto order parameter R

Usage:
    python -m sc_neurocore.experiments.demo_tcbo_consciousness

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import sys

import numpy as np

from .tcbo_demo_engine import TCBODemoEngine, SCENARIOS


def print_bar(value: float, width: int = 30, char: str = "█") -> str:
    """Render a simple ASCII progress bar."""
    filled = int(value * width)
    return char * filled + "░" * (width - filled)


def run_demo():
    """Run all 5 scenarios and display results."""
    engine = TCBODemoEngine(N=16, seed=42)

    print("=" * 72)
    print("  TCBO Consciousness Detection Demo")
    print("  Self-Consistent Phenomenological Network — SC-NeuroCore")
    print("=" * 72)
    print()

    for scenario_name, config in SCENARIOS.items():
        print(f"\n{'─' * 72}")
        print(f"  Scenario: {scenario_name.upper()}")
        print(f"  {config.description}")
        print(f"  Duration: {config.duration_steps} steps | Controller: {config.use_controller}")
        print(f"{'─' * 72}")
        print(f"  {'Step':>5} | {'R':>6} | {'p_h1':>6} | {'Gate':>5} | {'κ':>6} | Bar")
        print(f"  {'─' * 5} | {'─' * 6} | {'─' * 6} | {'─' * 5} | {'─' * 6} | {'─' * 30}")

        snapshots = engine.run_scenario(scenario_name)

        # Print every 10th step
        for snap in snapshots:
            if snap.tick % 10 == 0 or snap.tick == len(snapshots) - 1:
                gate_str = " OPEN" if snap.gate_open else "CLOSE"
                bar = print_bar(snap.p_h1)
                print(
                    f"  {snap.tick:5d} | {snap.R_global:6.3f} | {snap.p_h1:6.3f} | "
                    f"{gate_str} | {snap.kappa:6.3f} | {bar}"
                )

        # Summary
        p_h1_values = [s.p_h1 for s in snapshots]
        gate_changes = sum(
            1 for i in range(1, len(snapshots))
            if snapshots[i].gate_open != snapshots[i - 1].gate_open
        )
        print(f"\n  Summary:")
        print(f"    p_h1 range: [{min(p_h1_values):.3f}, {max(p_h1_values):.3f}]")
        print(f"    Gate transitions: {gate_changes}")
        print(f"    Final state: {'CONSCIOUS' if snapshots[-1].is_conscious else 'UNCONSCIOUS'}")

    print(f"\n{'=' * 72}")
    print("  All scenarios complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()

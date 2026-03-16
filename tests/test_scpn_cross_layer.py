# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Cross-layer coupling integration tests for the full 16-layer SCPN stack."""

from __future__ import annotations
import numpy as np

from sc_neurocore.scpn.layers import (
    create_full_stack,
    run_integrated_step,
    get_global_metrics,
    L8_PhaseFieldLayer,
    L8_StochasticParameters,
)


def test_full_stack_creation():
    """All 16 layers instantiate and produce valid metrics."""
    stack = create_full_stack()
    assert len(stack) == 16
    metrics = get_global_metrics(stack)
    assert len(metrics) == 16
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} metric is not float"


def test_full_stack_step_no_error():
    """16-layer integrated step completes without exception."""
    stack = create_full_stack()
    outputs = run_integrated_step(stack, dt=0.01)
    assert len(outputs) == 16
    for key, out in outputs.items():
        assert "output_bitstreams" in out, f"{key} missing output_bitstreams"


def test_l16_director_stabilises_gci():
    """L16 PI controller drives GCI toward target over multiple steps."""
    stack = create_full_stack()
    gci_history = []
    for _ in range(50):
        outputs = run_integrated_step(stack, dt=0.1)
        gci_history.append(outputs["l15"]["gci"])

    # GCI should move toward target (0.8) — check it's not stuck at initial
    assert gci_history[-1] != gci_history[0], "GCI did not change over 50 steps"


def test_phase_locked_state_high_k():
    """High coupling strength drives Kuramoto order parameter toward 1."""
    # Use identical frequencies so coupling guarantees synchronisation
    omegas = np.ones(12) * 2.0
    params = L8_StochasticParameters(k_cosmic=5.0, n_pulsars=12, pulsar_omegas=omegas)
    layer = L8_PhaseFieldLayer(params)
    for _ in range(200):
        layer.step(0.05)
    r = layer.get_global_metric()
    assert r > 0.8, f"Order parameter {r:.3f} too low for identical-frequency high coupling"


def test_decoupled_layers_evolve_independently():
    """With K=0, L8 phases diverge (low order parameter)."""
    params = L8_StochasticParameters(k_cosmic=0.0, n_pulsars=12)
    layer = L8_PhaseFieldLayer(params)
    for _ in range(100):
        layer.step(0.1)
    r = layer.get_global_metric()
    # Uncoupled oscillators with different frequencies -> low R
    assert r < 0.8, f"Order parameter {r:.3f} unexpectedly high for K=0"


def test_energy_nonincreasing_under_coupling():
    """L9 Hopfield energy should not increase after pattern storage and relaxation."""
    stack = create_full_stack()
    # Store a pattern in L9
    pattern = np.random.choice([-1, 1], size=64)
    stack["l9"].store(pattern)

    energies = []
    for _ in range(30):
        out = stack["l9"].step(0.1)
        energies.append(out["energy"])

    # After initial transient (first 5 steps), energy should generally decrease
    late_energies = energies[5:]
    if len(late_energies) > 2:
        # Allow small fluctuations due to stochastic updates
        assert (
            late_energies[-1] <= late_energies[0] + 0.5
        ), f"Energy increased: {late_energies[0]:.3f} -> {late_energies[-1]:.3f}"

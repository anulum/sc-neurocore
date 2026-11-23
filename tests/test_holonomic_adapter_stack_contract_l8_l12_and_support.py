# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (l8_l12_and_support) from former test_holonomic_adapter_stack_contract.py

"""Contract tests of the L8 to L12 adapters and their support modules."""

from __future__ import annotations

import numpy as np
from sc_neurocore.accel.jax_backend import (
    jnp,
)
from sc_neurocore.adapters.holonomic.dna_storage import (
    DNAEncoder,
)
from sc_neurocore.adapters.holonomic.grn import (
    GeneticRegulatoryLayer,
)
from sc_neurocore.adapters.holonomic.l10_fire import (
    L10_FirewallAdapter,
)
from sc_neurocore.adapters.holonomic.l11_noos import (
    L11_NoosphericAdapter,
)
from sc_neurocore.adapters.holonomic.l12_gaian import (
    L12_GaianAdapter,
)
from sc_neurocore.adapters.holonomic.l8_cosm import (
    L8_CosmicAdapter,
)
from sc_neurocore.adapters.holonomic.l9_mem import (
    L9_MemoryAdapter,
)
from sc_neurocore.adapters.holonomic.neuromodulation import (
    NeuromodulatorSystem,
)
from sc_neurocore.quantum.qec import (
    QecShield,
)


def test_l8_adapter_contract() -> None:
    """The L8 adapter steps and reports its locking index."""
    adapter = L8_CosmicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((12, 1024)))
    assert out.shape == (12, 1024)
    metrics = adapter.get_metrics()
    assert "pta_locking_index" in metrics


def test_l9_adapter_contract() -> None:
    """The L9 adapter steps and reports holographic overlap."""
    adapter = L9_MemoryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((64, 1024)))
    assert out.shape == (1024,)
    metrics = adapter.get_metrics()
    assert "holographic_overlap" in metrics


def test_l10_adapter_contract() -> None:
    """The L10 adapter steps and reports shielding potential."""
    adapter = L10_FirewallAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape == (100, 1024)
    metrics = adapter.get_metrics()
    assert "avg_shielding_potential" in metrics


def test_l11_adapter_contract() -> None:
    """The L11 adapter steps and reports entropy."""
    adapter = L11_NoosphericAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "noospheric_entropy" in metrics


def test_l12_adapter_contract() -> None:
    """The L12 adapter steps and reports ecosystem coherence."""
    adapter = L12_GaianAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "eco_system_coherence" in metrics


def test_qec_shield_contract() -> None:
    """The QEC shield encodes, finds syndromes and decodes back."""
    shield = QecShield(code_type="repetition", distance=3)
    bits = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    encoded = shield.encode(bits)
    assert encoded.shape == (2, 3, 2)
    syndromes = shield.extract_syndromes(encoded)
    assert syndromes.shape == (2, 2, 2)
    decoded = shield.decode(encoded)
    assert np.array_equal(decoded, bits)
    rate = shield.get_error_rate(syndromes)
    assert rate == 0.0


def test_dna_encoder_contract() -> None:
    """The DNA encoder round-trips eight bits through four bases."""
    encoder = DNAEncoder()
    # bitstream -> dna -> bitstream
    bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    dna = encoder.encode(bits)
    assert len(dna) == 4
    recovered = encoder.decode(dna)
    assert len(recovered) == 8
    assert np.array_equal(bits, recovered)


def test_grn_contract() -> None:
    """The gene network keeps one state per gene."""
    grn = GeneticRegulatoryLayer(n_neurons=5)
    grn.step(np.ones(5))
    state = grn.get_threshold_modulators()
    assert state.shape == (5,)


def test_l11_adapter_broadcasts_mismatched_input_width() -> None:
    """The L11 adapter broadcasts an input of another width."""
    # An informational drive whose node count differs from the adapter is
    # collapsed to a single mean and broadcast across all nodes.
    adapter = L11_NoosphericAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((50, 1024)))
    assert out.shape[0] == 100


def test_l12_adapter_broadcasts_mismatched_input_width() -> None:
    """The L12 adapter broadcasts an input of another width."""
    # The Gaian environmental drive applies the same broadcast rule.
    adapter = L12_GaianAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((50, 1024)))
    assert out.shape[0] == 100


def test_dna_encoder_pads_odd_length_bitstream() -> None:
    """The DNA encoder pads an odd bit count."""
    # An odd-length bitstream is zero-padded to an even length before the
    # two-bits-per-base mapping, so it still yields whole DNA bases.
    encoder = DNAEncoder()
    bits = np.array([1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    dna = encoder.encode(bits)
    assert len(dna) == 4


def test_neuromodulator_contract() -> None:
    """Neuromodulation lowers the threshold."""
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.8, stress=0.2)
    params = {"v_threshold": 1.0, "noise_std": 0.1}
    new_params = mod.modulate_neuron(params)
    assert new_params["v_threshold"] < 1.0


def test_neuromodulator_stress_clamp() -> None:
    """Serotonin stays within [0.1, 1] under stress."""
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.0, stress=10.0)
    assert mod.ht_level >= 0.1
    mod.update_levels(reward=0.0, stress=-10.0)
    assert mod.ht_level <= 1.0


def test_neuromodulator_no_matching_keys() -> None:
    """Parameters without modulated keys are unchanged."""
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.5, stress=0.5)
    params = {"some_other_param": 42}
    out = mod.modulate_neuron(params)
    assert out["some_other_param"] == 42

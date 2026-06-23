# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L2 neurochemical layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l2_neurochemical import (
    L2_NeurochemicalLayer,
    L2_StochasticParameters,
)


def test_l2_seed_scopes_receptor_and_output_bitstreams() -> None:
    params = L2_StochasticParameters(
        n_receptors=16,
        bitstream_length=128,
        rng_seed=123,
    )
    layer_a = L2_NeurochemicalLayer(params)
    layer_b = L2_NeurochemicalLayer(params)

    out_a0 = layer_a.step(0.5, nt_release=np.ones(4) * 0.2)["output_bitstreams"]
    out_b0 = layer_b.step(0.5, nt_release=np.ones(4) * 0.2)["output_bitstreams"]
    out_a1 = layer_a.step(0.5, nt_release=np.ones(4) * 0.2)["output_bitstreams"]
    out_b1 = layer_b.step(0.5, nt_release=np.ones(4) * 0.2)["output_bitstreams"]

    np.testing.assert_allclose(layer_a.receptor_states, layer_b.receptor_states)
    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l2_quantum_coupling_and_genomic_drive_are_validated_and_exported() -> None:
    params = L2_StochasticParameters(
        n_receptors=12,
        bitstream_length=16,
        quantum_coupling=0.25,
        genomic_coupling=0.5,
        rng_seed=456,
    )
    layer = L2_NeurochemicalLayer(params)
    layer.receptor_states[:] = 0.5

    result = layer.step(0.01, l1_input=np.ones(8))

    assert np.all(result["receptor_activity"] > 0.5)
    assert result["genomic_drive"].shape == (params.n_neurotransmitter_types,)
    np.testing.assert_allclose(
        result["genomic_drive"], params.genomic_coupling * result["second_messengers"]
    )


def test_l2_release_neurotransmitter_validates_indices_and_amounts() -> None:
    layer = L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=4, bitstream_length=16))

    before = layer.nt_concentrations.copy()
    layer.release_neurotransmitter(layer.DA, 0.25)

    assert layer.nt_concentrations[layer.DA] > before[layer.DA]
    with pytest.raises(ValueError, match="nt_type"):
        layer.release_neurotransmitter(99, 0.1)
    with pytest.raises(ValueError, match="amount"):
        layer.release_neurotransmitter(layer.DA, np.nan)
    with pytest.raises(ValueError, match="amount"):
        layer.release_neurotransmitter(layer.DA, -0.1)


def test_l2_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_receptors"):
        L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=0))
    with pytest.raises(ValueError, match="n_neurotransmitter_types"):
        L2_NeurochemicalLayer(L2_StochasticParameters(n_neurotransmitter_types=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L2_NeurochemicalLayer(L2_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="binding_affinity"):
        L2_NeurochemicalLayer(L2_StochasticParameters(binding_affinity=1.1))
    with pytest.raises(ValueError, match="unbinding_rate"):
        L2_NeurochemicalLayer(L2_StochasticParameters(unbinding_rate=-0.1))
    with pytest.raises(ValueError, match="diffusion_rate"):
        L2_NeurochemicalLayer(L2_StochasticParameters(diffusion_rate=-0.1))
    with pytest.raises(ValueError, match="reuptake_rate"):
        L2_NeurochemicalLayer(L2_StochasticParameters(reuptake_rate=-0.1))
    with pytest.raises(ValueError, match="quantum_coupling"):
        L2_NeurochemicalLayer(L2_StochasticParameters(quantum_coupling=-0.1))
    with pytest.raises(ValueError, match="genomic_coupling"):
        L2_NeurochemicalLayer(L2_StochasticParameters(genomic_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L2_NeurochemicalLayer(L2_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=4, bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="nt_release"):
        layer.step(0.01, nt_release=np.ones(3))
    with pytest.raises(ValueError, match="nt_release"):
        layer.step(0.01, nt_release=np.array([0.0, 0.0, 0.0, np.nan]))
    with pytest.raises(ValueError, match="l1_input"):
        layer.step(0.01, l1_input=np.array([0.5, np.nan]))


def test_l2_history_is_trimmed_past_window() -> None:
    layer = L2_NeurochemicalLayer(
        L2_StochasticParameters(n_receptors=4, bitstream_length=16, rng_seed=3)
    )
    for _ in range(110):
        layer.step(0.01)
    # The rolling history is capped at the 100-step window.
    assert len(layer.history) <= 100


def test_l2_release_neurotransmitter_rejects_non_integer_type() -> None:
    layer = L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=4, bitstream_length=16))
    with pytest.raises(ValueError, match="nt_type must be a valid neurotransmitter index"):
        layer.release_neurotransmitter(cast(int, True), 0.1)


def test_l2_state_accessors_and_negative_seed() -> None:
    layer = L2_NeurochemicalLayer(
        L2_StochasticParameters(n_receptors=4, bitstream_length=16, rng_seed=2)
    )
    assert 0.0 <= layer.get_global_metric() <= 1.0
    assert "dopamine" in layer.get_neuromodulation_state()
    with pytest.raises(ValueError, match="rng_seed"):
        L2_NeurochemicalLayer(L2_StochasticParameters(rng_seed=-1))


def test_l2_nt_release_out_of_range_rejected() -> None:
    layer = L2_NeurochemicalLayer(L2_StochasticParameters(n_receptors=4, bitstream_length=16))
    with pytest.raises(ValueError, match=r"nt_release must be within \[0, 1\]"):
        layer.step(0.01, nt_release=np.array([0.0, 0.0, 0.0, 1.5]))

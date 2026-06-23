# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L3 genomic layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l3_genomic import L3_GenomicLayer, L3_StochasticParameters


def test_l3_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L3_StochasticParameters(
        n_genes=12,
        n_regulatory_elements=4,
        bitstream_length=128,
        rng_seed=123,
    )
    layer_a = L3_GenomicLayer(params)
    layer_b = L3_GenomicLayer(params)

    np.testing.assert_allclose(layer_a.expression_levels, layer_b.expression_levels)
    np.testing.assert_allclose(layer_a.protein_levels, layer_b.protein_levels)
    np.testing.assert_array_equal(layer_a.chromatin_state, layer_b.chromatin_state)
    np.testing.assert_allclose(layer_a.regulatory_matrix, layer_b.regulatory_matrix)
    out_a0 = layer_a.step(0.01)["output_bitstreams"]
    out_b0 = layer_b.step(0.01)["output_bitstreams"]
    out_a1 = layer_a.step(0.01)["output_bitstreams"]
    out_b1 = layer_b.step(0.01)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l3_ciss_and_cellular_drive_are_vectorized_and_wired() -> None:
    params = L3_StochasticParameters(
        n_genes=10,
        n_regulatory_elements=4,
        bitstream_length=16,
        ciss_efficiency=0.5,
        dna_chirality=1.0,
        cellular_coupling=0.25,
        rng_seed=456,
    )
    layer = L3_GenomicLayer(params)

    result = layer.step(0.01)

    assert result["spin_polarization"].shape == (params.n_genes,)
    assert result["cellular_drive"].shape == (params.n_genes,)
    np.testing.assert_allclose(
        result["cellular_drive"], params.cellular_coupling * result["protein_levels"]
    )


def test_l3_neurochemical_and_bioelectric_inputs_are_validated_and_used() -> None:
    layer = L3_GenomicLayer(
        L3_StochasticParameters(
            n_genes=8,
            n_regulatory_elements=4,
            bitstream_length=16,
            rng_seed=789,
        )
    )

    before_expression = layer.expression_levels.copy()
    before_membrane = layer.membrane_potential.copy()
    result = layer.step(
        0.01,
        l2_input={"second_messengers": np.ones(8)},
        bioelectric_signal=np.linspace(-68.0, -66.0, 8),
    )

    assert np.mean(result["expression_levels"]) > np.mean(before_expression)
    assert not np.array_equal(result["membrane_potential"], before_membrane)

    broadcast = layer.step(0.01, bioelectric_signal=np.array([-65.0]))
    assert broadcast["membrane_potential"].shape == (8,)


def test_l3_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_genes"):
        L3_GenomicLayer(L3_StochasticParameters(n_genes=0))
    with pytest.raises(ValueError, match="n_regulatory_elements"):
        L3_GenomicLayer(L3_StochasticParameters(n_regulatory_elements=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L3_GenomicLayer(L3_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="transcription_rate"):
        L3_GenomicLayer(L3_StochasticParameters(transcription_rate=-0.1))
    with pytest.raises(ValueError, match="translation_rate"):
        L3_GenomicLayer(L3_StochasticParameters(translation_rate=-0.1))
    with pytest.raises(ValueError, match="degradation_rate"):
        L3_GenomicLayer(L3_StochasticParameters(degradation_rate=-0.1))
    with pytest.raises(ValueError, match="ciss_efficiency"):
        L3_GenomicLayer(L3_StochasticParameters(ciss_efficiency=1.1))
    with pytest.raises(ValueError, match="dna_chirality"):
        L3_GenomicLayer(L3_StochasticParameters(dna_chirality=0.0))
    with pytest.raises(ValueError, match="methylation_rate"):
        L3_GenomicLayer(L3_StochasticParameters(methylation_rate=-0.1))
    with pytest.raises(ValueError, match="demethylation_rate"):
        L3_GenomicLayer(L3_StochasticParameters(demethylation_rate=-0.1))
    with pytest.raises(ValueError, match="histone_mod_rate"):
        L3_GenomicLayer(L3_StochasticParameters(histone_mod_rate=-0.1))
    with pytest.raises(ValueError, match="bioelectric_coupling"):
        L3_GenomicLayer(L3_StochasticParameters(bioelectric_coupling=-0.1))
    with pytest.raises(ValueError, match="membrane_potential_rest"):
        L3_GenomicLayer(L3_StochasticParameters(membrane_potential_rest=np.nan))
    with pytest.raises(ValueError, match="neurochemical_coupling"):
        L3_GenomicLayer(L3_StochasticParameters(neurochemical_coupling=-0.1))
    with pytest.raises(ValueError, match="cellular_coupling"):
        L3_GenomicLayer(L3_StochasticParameters(cellular_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L3_GenomicLayer(L3_StochasticParameters(rng_seed=cast(Any, 1.5)))
    with pytest.raises(ValueError, match="rng_seed"):
        L3_GenomicLayer(L3_StochasticParameters(rng_seed=-1))

    layer = L3_GenomicLayer(
        L3_StochasticParameters(n_genes=4, n_regulatory_elements=2, bitstream_length=16)
    )
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="second_messengers"):
        layer.step(0.01, l2_input={"second_messengers": np.array([1.0, np.nan])})
    with pytest.raises(ValueError, match="bioelectric_signal"):
        layer.step(0.01, bioelectric_signal=np.ones(3))
    with pytest.raises(ValueError, match="bioelectric_signal"):
        layer.step(0.01, bioelectric_signal=np.array([-70.0, -70.0, -70.0, np.nan]))


def test_l3_global_metric_and_ciss_coherence_accessors() -> None:
    layer = L3_GenomicLayer(
        L3_StochasticParameters(n_genes=12, n_regulatory_elements=4, rng_seed=7)
    )
    assert 0.0 <= layer.get_global_metric() <= 1.0
    assert layer.get_ciss_coherence() >= 0.0

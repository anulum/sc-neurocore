# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision planner contracts

"""Focused adaptive precision planner contracts."""

from __future__ import annotations


import numpy as np

from sc_neurocore.compiler.adaptive_precision import (
    SynapsePrecision,
    assign_synapse_precisions,
)


def test_zero_sensitivity_synapses_use_minimum_length_and_zero_stochastic_bound() -> None:
    """Zero-sensitivity synapses keep the minimum bitstream length."""
    [row] = assign_synapse_precisions(
        [np.array([[0.0]])],
        min_length=16,
        max_length=128,
    )

    assert row.bitstream_length == 16
    assert row.sensitivity == 0.0
    assert row.stochastic_error_bound == 0.0


def test_assign_synapse_precisions_supports_one_dimensional_sensitivity_maps() -> None:
    """One-dimensional synapse plans preserve vector coordinates in manifest rows."""
    rows = assign_synapse_precisions(
        [np.array([0.25, 0.75])],
        layer_names=["vector"],
        sensitivity_maps=[np.array([0.1, 1.0])],
        target_error=0.05,
        min_bits=2,
        max_bits=8,
        min_length=16,
        max_length=512,
    )

    assert len(rows) == 2
    assert all(isinstance(row, SynapsePrecision) for row in rows)
    assert [row.to_dict()["input_index"] for row in rows] == [0, 1]
    assert {row.layer_name for row in rows} == {"vector"}
    assert rows[1].sensitivity > rows[0].sensitivity

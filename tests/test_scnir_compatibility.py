# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR compatibility matrix

"""Contract tests for the executable SC-NIR compatibility matrix."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nir")

from sc_neurocore.ir import (
    scnir_compatibility_matrix,
    scnir_compatibility_matrix_dicts,
    validate_scnir_compatibility_matrix,
)
from sc_neurocore.nir_bridge.node_map import NODE_MAP


def test_scnir_compatibility_matrix_covers_parser_primitives() -> None:
    validate_scnir_compatibility_matrix()

    primitives = {row.nir_primitive for row in scnir_compatibility_matrix()}
    assert {primitive.__name__ for primitive in NODE_MAP}.issubset(primitives)
    assert "NIRGraph" in primitives


def test_scnir_compatibility_matrix_is_deterministic_json() -> None:
    left = scnir_compatibility_matrix_dicts()
    right = scnir_compatibility_matrix_dicts()

    assert left == right
    assert json.loads(json.dumps(left, sort_keys=True))[0]["nir_primitive"] == "Input"


def test_scnir_compatibility_matrix_marks_closed_hdl_population_rows() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    lif = rows["LIF"]
    assert lif.support_level == "metadata_and_hdl"
    assert "signal_kind=spike" in lif.scnir_stream_metadata
    assert "encoding=unipolar" in lif.scnir_stream_metadata
    assert "lfsr16" in lif.source_metadata
    assert "sobol16" in lif.source_metadata

    li = rows["LI"]
    assert li.support_level == "metadata_and_hdl"
    assert "signal_kind=analogue_state" in li.scnir_stream_metadata
    assert "direct analogue-state MAC" in li.hdl_support


def test_scnir_compatibility_matrix_does_not_overclaim_parser_only_rows() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    conv1d = rows["Conv1d"]
    assert conv1d.support_level == "metadata_and_hdl"
    assert "convolution_lowered_weight" in conv1d.scnir_stream_metadata
    assert "dense Toeplitz" in conv1d.hdl_support

    conv2d = rows["Conv2d"]
    assert conv2d.support_level == "parser_only"
    assert conv2d.scnir_stream_metadata == ()
    assert conv2d.hdl_support != ""

    scale = rows["Scale"]
    assert scale.support_level == "metadata_and_hdl"
    assert "folded_weight_scale" in scale.scnir_stream_metadata
    assert "folded fixed-point gain" in scale.hdl_support

    flatten = rows["Flatten"]
    assert flatten.support_level == "metadata_and_hdl"
    assert "shape_preserving_flatten" in flatten.scnir_stream_metadata
    assert "fixed-point weight indexing" in flatten.hdl_support

    threshold = rows["Threshold"]
    assert threshold.support_level == "metadata_and_hdl"
    assert "threshold_transform" in threshold.scnir_stream_metadata
    assert "fixed-point comparator" in threshold.hdl_support

    delay = rows["Delay"]
    assert delay.support_level == "metadata_and_hdl"
    assert "delay_steps>=0" in delay.scnir_stream_metadata
    assert "register chain" in delay.hdl_support

    integrator = rows["I"]
    assert integrator.support_level == "metadata_and_hdl"
    assert "signal_kind=analogue_state" in integrator.scnir_stream_metadata
    assert "integrator state-update module" in integrator.hdl_support


def test_scnir_compatibility_matrix_records_weight_and_recurrent_delay_semantics() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    affine = rows["Affine"]
    assert affine.support_level == "metadata_and_hdl"
    assert "signal_kind=weight" in affine.scnir_stream_metadata
    assert "encoding=bipolar" in affine.scnir_stream_metadata

    linear = rows["Linear"]
    assert "delay_steps=0_or_1" in linear.scnir_stream_metadata
    assert "recurrent unit-delay" in linear.limitation

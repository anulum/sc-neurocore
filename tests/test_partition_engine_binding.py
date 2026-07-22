# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — hierarchical partition engine-binding contracts

"""Installed-extension contracts for Kernighan-Lin partition refinement."""

from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _two_vertex_inputs() -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
]:
    return (
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([1, 0], dtype=np.int32),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int64),
    )


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_kl_refine

    assert function.__name__ == "py_kl_refine"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(adj_offsets, adj_neighbours, adj_scc_abs, vertex_weights, part_map, "
        "parts_concat, parts_offsets, n_parts, kl_iterations, correlation_penalty)"
    )
    assert engine.py_kl_refine is function


def test_direct_binding_preserves_partition_and_input_ownership() -> None:
    inputs = _two_vertex_inputs()
    original_part_map = inputs[4].copy()

    part_map, moves = extension.py_kl_refine(*inputs, 2, 1, 0.0)

    np.testing.assert_array_equal(part_map, [0, 1])
    np.testing.assert_array_equal(inputs[4], original_part_map)
    assert part_map.dtype == np.int32
    assert moves == 0


def test_zero_iterations_is_a_stable_no_op() -> None:
    inputs = _two_vertex_inputs()

    part_map, moves = extension.py_kl_refine(*inputs, 2, 0, 0.5)

    np.testing.assert_array_equal(part_map, inputs[4])
    assert moves == 0

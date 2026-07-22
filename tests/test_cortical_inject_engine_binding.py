# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cortical injection engine-binding contracts

"""Installed-extension contracts for cortical-column CSR injection."""

from __future__ import annotations

import importlib

import numpy as np

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_names_signatures_and_top_level_identities_are_stable() -> None:
    single = extension.py_parallel_csr_spmv_add
    multiple = extension.py_parallel_csr_multi_spmv_add

    assert single.__name__ == "py_parallel_csr_spmv_add"
    assert single.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert single.__text_signature__ == "(indptr, indices, data, x, y)"
    assert engine.py_parallel_csr_spmv_add is single

    assert multiple.__name__ == "py_parallel_csr_multi_spmv_add"
    assert multiple.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert multiple.__text_signature__ == "(indptrs, indices_list, data_list, xs, y)"
    assert engine.py_parallel_csr_multi_spmv_add is multiple


def test_single_block_adds_csr_product_in_place() -> None:
    indptr = np.asarray([0, 2, 3], dtype=np.int32)
    indices = np.asarray([0, 1, 1], dtype=np.int32)
    data = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    source = np.asarray([4.0, 5.0], dtype=np.float64)
    output = np.asarray([10.0, 20.0], dtype=np.float64)

    result = extension.py_parallel_csr_spmv_add(indptr, indices, data, source, output)

    assert result is None
    np.testing.assert_array_equal(output, [24.0, 35.0])


def test_multi_block_accumulates_every_product_in_one_call() -> None:
    indptrs = [
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
    ]
    indices = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1, 0], dtype=np.int32),
    ]
    data = [
        np.asarray([2.0, 3.0], dtype=np.float64),
        np.asarray([4.0, 5.0], dtype=np.float64),
    ]
    sources = [
        np.asarray([7.0, 11.0], dtype=np.float64),
        np.asarray([13.0, 17.0], dtype=np.float64),
    ]
    output = np.asarray([1.0, 2.0], dtype=np.float64)

    result = extension.py_parallel_csr_multi_spmv_add(
        indptrs,
        indices,
        data,
        sources,
        output,
    )

    assert result is None
    np.testing.assert_array_equal(output, [83.0, 100.0])

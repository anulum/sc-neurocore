# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bitstream engine-binding contracts

"""Installed-extension contracts for packed bitstream operations."""

from __future__ import annotations

import importlib

import numpy as np

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_names_signatures_and_top_level_identities_are_stable() -> None:
    signatures = {
        "pack_bitstream": "(bits)",
        "unpack_bitstream": "(packed, original_length, original_shape=None)",
        "pack_bitstream_numpy": "(bits)",
        "popcount_numpy": "(packed)",
        "unpack_bitstream_numpy": "(packed, original_length)",
        "batch_encode": "(probs, length=1024, seed=44257)",
        "batch_encode_numpy": "(probs, length=1024, seed=44257)",
    }

    for name, signature in signatures.items():
        function = getattr(extension, name)
        assert function.__name__ == name
        assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
        assert function.__text_signature__ == signature
        assert getattr(engine, name) is function

    class_signatures = {
        "Lfsr16": "(seed=44257)",
        "BitstreamEncoder": "(data_width=16, seed=44257)",
        "BitstreamAverager": "(window=1024)",
    }
    for name, signature in class_signatures.items():
        binding_class = getattr(extension, name)
        assert binding_class.__name__ == name
        assert binding_class.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
        assert binding_class.__text_signature__ == signature
        assert getattr(engine, name) is binding_class


def test_bitstream_averager_preserves_windowed_estimate_and_reset() -> None:
    averager = extension.BitstreamAverager(window=4)

    for bit in (1, 0, 1, 1):
        averager.push(bit)

    assert averager.window == 4
    assert averager.estimate() == 0.75
    averager.reset()
    assert averager.estimate() == 0.0


def test_generic_one_and_two_dimensional_roundtrips_are_exact() -> None:
    flat = [1, 0, 1, 1, 0, 0, 1]
    packed_flat = extension.pack_bitstream(flat)
    assert extension.unpack_bitstream(packed_flat, len(flat)) == bytes(flat)

    rows = [flat, list(reversed(flat))]
    packed_rows = extension.pack_bitstream(rows)
    assert extension.unpack_bitstream(
        packed_rows,
        2 * len(flat),
        original_shape=(2, len(flat)),
    ) == [bytes(row) for row in rows]


def test_numpy_roundtrip_and_popcount_preserve_bits() -> None:
    bits = np.asarray(([1, 0, 1, 1, 0, 0, 1] * 10)[:67], dtype=np.uint8)

    packed = extension.pack_bitstream_numpy(bits)
    restored = extension.unpack_bitstream_numpy(packed, len(bits))

    np.testing.assert_array_equal(restored, bits)
    assert extension.popcount_numpy(packed) == int(bits.sum())


def test_batch_encoders_are_seed_deterministic_and_shape_stable() -> None:
    probabilities = np.asarray([0.0, 0.25, 1.0], dtype=np.float64)

    generic_a = extension.batch_encode(probabilities, length=65, seed=17)
    generic_b = extension.batch_encode(probabilities, length=65, seed=17)
    numpy_a = extension.batch_encode_numpy(probabilities, length=65, seed=17)
    numpy_b = extension.batch_encode_numpy(probabilities, length=65, seed=17)

    assert generic_a == generic_b
    assert len(generic_a) == 3
    assert all(len(row) == 2 for row in generic_a)
    np.testing.assert_array_equal(numpy_a, numpy_b)
    assert numpy_a.shape == (3, 2)

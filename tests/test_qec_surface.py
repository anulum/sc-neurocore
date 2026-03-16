# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SurfaceCodeShield (d=3 rotated surface code)

"""Tests for SurfaceCodeShield (d=3 rotated surface code)."""

from __future__ import annotations
import numpy as np
import pytest

from sc_neurocore.quantum.qec import SurfaceCodeShield


@pytest.fixture
def shield():
    return SurfaceCodeShield(distance=3)


def test_init_distance_validation():
    with pytest.raises(ValueError):
        SurfaceCodeShield(distance=2)
    with pytest.raises(ValueError):
        SurfaceCodeShield(distance=4)


def test_encode_shape(shield):
    logical = np.array([[1, 0, 1, 1, 0]], dtype=np.uint8)
    physical = shield.encode(logical)
    assert physical.shape == (1, 9, 5)
    np.testing.assert_array_equal(physical[0, 0, :], logical[0])
    np.testing.assert_array_equal(physical[0, 8, :], logical[0])


def test_encode_decode_roundtrip_no_errors(shield):
    rng = np.random.default_rng(42)
    logical = rng.integers(0, 2, size=(4, 64), dtype=np.uint8)
    physical = shield.encode(logical)
    decoded = shield.decode(physical)
    np.testing.assert_array_equal(decoded, logical)


def test_single_x_error_correction(shield):
    """Flip one data qubit (bit-flip) per position — should be corrected."""
    rng = np.random.default_rng(7)
    logical = rng.integers(0, 2, size=(1, 32), dtype=np.uint8)
    physical = shield.encode(logical)
    for t in range(physical.shape[2]):
        qubit = rng.integers(0, 9)
        physical[0, qubit, t] ^= 1
    decoded = shield.decode(physical)
    np.testing.assert_array_equal(decoded, logical)


def test_single_z_error_correction(shield):
    """Z error on a surface code is a phase flip — same as X in classical bits."""
    rng = np.random.default_rng(13)
    logical = rng.integers(0, 2, size=(2, 16), dtype=np.uint8)
    physical = shield.encode(logical)
    for l_idx in range(2):
        for t in range(16):
            qubit = rng.integers(0, 9)
            physical[l_idx, qubit, t] ^= 1
    decoded = shield.decode(physical)
    np.testing.assert_array_equal(decoded, logical)


def test_syndrome_zero_for_clean_codeword(shield):
    logical = np.ones((1, 8), dtype=np.uint8)
    physical = shield.encode(logical)
    x_syn, z_syn = shield.measure_syndrome(physical)
    assert np.all(x_syn == 0)
    assert np.all(z_syn == 0)


def test_syndrome_nonzero_for_error(shield):
    logical = np.zeros((1, 8), dtype=np.uint8)
    physical = shield.encode(logical)
    physical[0, 4, 0] = 1  # flip center qubit
    x_syn, z_syn = shield.measure_syndrome(physical)
    assert np.any(x_syn[:, :, 0] != 0) or np.any(z_syn[:, :, 0] != 0)


def test_error_rate_computation(shield):
    logical = np.zeros((1, 100), dtype=np.uint8)
    physical = shield.encode(logical)
    x_syn, z_syn = shield.measure_syndrome(physical)
    assert shield.get_error_rate(x_syn, z_syn) == 0.0


def test_d5_creation():
    s = SurfaceCodeShield(distance=5)
    assert s.n_data == 25
    logical = np.ones((1, 4), dtype=np.uint8)
    physical = s.encode(logical)
    assert physical.shape == (1, 25, 4)
    decoded = s.decode(physical)
    np.testing.assert_array_equal(decoded, logical)

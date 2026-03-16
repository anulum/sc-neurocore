# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for memory estimation utilities."""

from __future__ import annotations

import numpy as np
from sc_neurocore.utils.profiling import estimate_memory


class _FakeLayer:
    """Minimal stub matching the SC layer interface."""

    def __init__(self, n_in, n_out, length=256):
        self.weights = np.zeros((n_out, n_in), dtype=np.float64)
        self.length = length


def test_single_layer_bytes():
    layer = _FakeLayer(50, 128, length=256)
    result = estimate_memory([layer], unit="B")

    assert result["weights_bytes"] == 128 * 50 * 8  # float64
    words_per_weight = int(np.ceil(256 / 64))  # 4
    assert result["packed_bytes"] == 128 * 50 * words_per_weight * 8
    assert result["neuron_state_bytes"] == 128 * 9
    assert result["total_bytes"] == (
        result["weights_bytes"] + result["packed_bytes"] + result["neuron_state_bytes"]
    )


def test_multiple_layers():
    layers = [_FakeLayer(10, 20), _FakeLayer(20, 5)]
    result = estimate_memory(layers, unit="B")
    assert result["weights_bytes"] == (20 * 10 + 5 * 20) * 8
    assert result["neuron_state_bytes"] == (20 + 5) * 9


def test_unit_mb():
    layer = _FakeLayer(100, 100, length=1024)
    result = estimate_memory([layer], unit="MB")
    assert "MB" in result["total_human"]
    assert float(result["total_human"].split()[0]) > 0


def test_unit_kb():
    layer = _FakeLayer(10, 10, length=64)
    result = estimate_memory([layer], unit="KB")
    assert "KB" in result["total_human"]


def test_empty_layer_list():
    result = estimate_memory([], unit="B")
    assert result["total_bytes"] == 0


def test_layer_without_weights():
    class Bare:
        length = 128

    result = estimate_memory([Bare()], unit="B")
    assert result["total_bytes"] == 0


def test_top_level_import():
    from sc_neurocore import estimate_memory as em

    assert em is estimate_memory

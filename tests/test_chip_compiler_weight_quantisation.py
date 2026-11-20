# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight quantisation of the chip compiler at every precision

"""Hold the chip compiler's weight quantisation to finite weights on every target precision.

A 1-bit target once divided by zero: the scale became infinite, every quantised weight NaN, and the result still said
``success=True`` with no warning. These tests hold the binary rule for one bit, the refusals for a precision below one
bit and for weights that are empty or not finite, and the report lines for violations and warnings.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.chip_compiler import ChipSpec, CoreSpec, compile_for_chip


def _chip(weight_bits: int) -> ChipSpec:
    """Return a small all-digital chip whose cores hold weights of ``weight_bits`` bits."""
    return ChipSpec(
        name=f"test_{weight_bits}bit",
        vendor="Test",
        total_cores=4,
        core=CoreSpec(
            max_neurons=64,
            max_synapses_per_neuron=256,
            weight_bits=weight_bits,
            supported_neuron_types=["LIF"],
        ),
    )


def test_one_bit_weights_take_two_signed_levels_of_the_mean_magnitude() -> None:
    """One bit codes a sign: every weight becomes plus or minus the mean magnitude, and zero takes the plus sign."""
    w = np.array([[0.5, -0.25, 0.0], [1.0, -2.0, 0.75]])
    result = compile_for_chip([(3, 2)], weights=[w], target=_chip(1))
    assert result.success, result.violations
    (q,) = result.quantized_weights
    level = float(np.mean(np.abs(w)))
    assert np.all(np.isfinite(q))
    np.testing.assert_array_equal(q, np.where(w >= 0, level, -level))
    assert set(np.unique(q).tolist()) == {-level, level}


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_multi_bit_weights_stay_finite_on_the_symmetric_grid(bits: int) -> None:
    """With ``b`` bits the levels are whole multiples of max|w| / (2**(b-1) - 1), within max|w|."""
    w = np.random.default_rng(bits).normal(size=(8, 16))
    result = compile_for_chip([(16, 8)], weights=[w], target=_chip(bits))
    assert result.success, result.violations
    (q,) = result.quantized_weights
    abs_max = float(np.abs(w).max())
    step = abs_max / (2 ** (bits - 1) - 1)
    assert np.all(np.isfinite(q))
    assert float(np.abs(q).max()) <= abs_max
    np.testing.assert_allclose(q / step, np.round(q / step), atol=1e-9)


def test_a_supported_neuron_type_adds_no_violation() -> None:
    """A layer whose neuron type the core supports compiles; only the unsupported one is reported."""
    ok = compile_for_chip([(4, 4)], neuron_types=["LIF"], target=_chip(8))
    assert ok.success and ok.violations == []
    mixed = compile_for_chip([(4, 4), (4, 4)], neuron_types=["LIF", "AdEx"], target=_chip(8))
    assert [v.split(":")[0] for v in mixed.violations] == ["Layer 1"]


def test_a_precision_below_one_bit_is_a_violation() -> None:
    """A core spec with zero weight bits cannot hold a weight: the compilation fails and quantises nothing."""
    result = compile_for_chip([(2, 2)], weights=[np.ones((2, 2))], target=_chip(0))
    assert not result.success
    assert any("weight precision" in v for v in result.violations)
    assert result.quantized_weights == []


@pytest.mark.parametrize(
    ("bad", "word"),
    [
        (np.array([[0.5, np.nan]]), "not finite"),
        (np.array([[np.inf, 0.5]]), "not finite"),
        (np.zeros((0, 3)), "empty"),
    ],
)
def test_weights_that_are_empty_or_not_finite_are_a_violation(bad: np.ndarray, word: str) -> None:
    """No layer is quantised when one weight matrix is empty or holds NaN or infinity, and the result says why."""
    good = np.ones((2, 2))
    result = compile_for_chip([(2, 2), (2, 2)], weights=[good, bad], target=_chip(8))
    assert not result.success
    assert any(word in v and "layer 1" in v for v in result.violations), result.violations
    assert result.quantized_weights == []


def test_precision_loss_is_warned_and_the_summary_lists_warnings_and_violations() -> None:
    """A coarse grid warns about its relative error; the summary prints each warning and each violation."""
    w = np.random.default_rng(0).normal(size=(8, 16))
    coarse = compile_for_chip([(16, 8)], weights=[w], target=_chip(2))
    assert coarse.success
    assert any("relative error" in warning for warning in coarse.warnings)
    assert "[WARNING] Weight quantisation to 2-bit" in coarse.summary()

    failed = compile_for_chip([(2, 2)], weights=[np.ones((2, 2))], target=_chip(0))
    report = failed.summary()
    assert "FAILED" in report
    assert "[VIOLATION]" in report and "weight precision" in report

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shipped spike-train statistics core: values, edges, re-exports, parity

"""Tests for :mod:`sc_neurocore.spike_train_statistics`.

The core ships in the wheel and :mod:`sc_neurocore.analysis.spike_stats`
re-exports it, so these tests pin the values on known trains, every edge case,
that the source-checkout names are the same objects, that the core imports
nothing from ``sc_neurocore.analysis``, and that permutation entropy agrees
with an independent loop implementation and with the Rust helper when built.
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import sc_neurocore.spike_train_statistics as core
from sc_neurocore import analysis
from sc_neurocore.analysis.spike_stats import basic as analysis_basic
from sc_neurocore.analysis.spike_stats import spectral as analysis_spectral
from sc_neurocore.analysis.spike_stats import variability as analysis_variability

_CORE_DIR = Path(core.__file__).resolve().parent


def _train(*steps: int, length: int) -> np.ndarray[Any, Any]:
    """Return a binary train of ``length`` steps with spikes at ``steps``."""
    train = np.zeros(length, dtype=np.int8)
    train[list(steps)] = 1
    return train


def _loop_permutation_entropy(values: np.ndarray[Any, Any], order: int, delay: int) -> float:
    """Compute the entropy one window at a time, ties ranked by position, as an oracle."""
    x = [float(v) for v in values]
    n_patterns = len(x) - (order - 1) * delay
    counts: dict[tuple[int, ...], int] = {}
    for start in range(n_patterns):
        window = [x[start + j * delay] for j in range(order)]
        by_value = sorted(range(order), key=lambda j: (window[j], j))
        rank = [0] * order
        for position, index in enumerate(by_value):
            rank[index] = position
        counts[tuple(rank)] = counts.get(tuple(rank), 0) + 1
    h = -sum(c / n_patterns * math.log2(c / n_patterns) for c in counts.values())
    return h / math.log2(math.factorial(order))


def test_public_surface_is_the_nine_promoted_functions() -> None:
    """The package exports exactly the bounded surface it promises."""
    assert sorted(core.__all__) == [
        "bin_spike_train",
        "cv_isi",
        "fano_factor",
        "firing_rate",
        "isi",
        "permutation_entropy",
        "power_spectrum",
        "spike_count",
        "spike_times",
    ]


def test_core_imports_nothing_from_analysis() -> None:
    """No module of the core names ``sc_neurocore.analysis`` in an import."""
    imported: list[str] = []
    for path in sorted(_CORE_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
    assert imported, "the scan found no imports at all"
    assert "numpy" in imported
    assert not [name for name in imported if name.startswith("sc_neurocore.analysis")]


@pytest.mark.parametrize(
    "name", ["bin_spike_train", "firing_rate", "isi", "spike_count", "spike_times"]
)
def test_analysis_basic_names_are_the_core_objects(name: str) -> None:
    """The source-checkout basic module re-exports the core functions themselves."""
    assert getattr(analysis_basic, name) is getattr(core, name)
    assert getattr(analysis, name) is getattr(core, name)


def test_analysis_spectral_and_variability_names_are_the_core_objects() -> None:
    """Power spectrum, ISI CV and Fano factor are the core functions under the old names."""
    assert analysis_spectral.power_spectrum is core.power_spectrum
    assert analysis.power_spectrum is core.power_spectrum
    for name in ("cv_isi", "fano_factor"):
        assert getattr(analysis_variability, name) is getattr(core, name)
        assert getattr(analysis, name) is getattr(core, name)


def test_basic_operations_on_a_known_train() -> None:
    """Spike times, intervals, rate, count and bins of a hand-built train."""
    train = _train(1, 4, 9, length=12)
    np.testing.assert_allclose(core.spike_times(train, dt=0.01), [0.01, 0.04, 0.09])
    np.testing.assert_allclose(core.isi(train, dt=0.01), [0.03, 0.05])
    assert core.firing_rate(train, dt=0.01) == pytest.approx(3 / 0.12)
    assert core.spike_count(train) == 3
    np.testing.assert_array_equal(core.bin_spike_train(train, bin_size=5), [2, 1])


def test_basic_operations_on_degenerate_trains() -> None:
    """Fewer than two spikes, zero duration, and a train shorter than one bin."""
    one_spike = _train(2, length=5)
    intervals = core.isi(one_spike)
    assert intervals.size == 0 and intervals.dtype == np.float64
    assert core.firing_rate(np.zeros(0, dtype=np.int8)) == 0.0
    assert core.firing_rate(one_spike, dt=0.0) == 0.0
    np.testing.assert_array_equal(core.bin_spike_train(one_spike, bin_size=10), [1])


def test_power_spectrum_of_a_periodic_train_lies_on_its_harmonics() -> None:
    """A spike every 10 ms has power only at multiples of 100 Hz, 100**2 / 1000 each."""
    train = np.zeros(1000, dtype=np.int8)
    train[::10] = 1
    psd, freqs = core.power_spectrum(train, dt=0.001)
    assert psd.shape == freqs.shape == (501,)
    assert freqs[1] == pytest.approx(1.0)
    lines = np.flatnonzero(psd > 1e-9)
    np.testing.assert_allclose(freqs[lines], [100.0, 200.0, 300.0, 400.0, 500.0])
    np.testing.assert_allclose(psd[lines], 10.0)
    assert psd[0] == pytest.approx(0.0, abs=1e-12)


def test_power_spectrum_of_a_single_sample_is_empty() -> None:
    """Fewer than two samples give two empty arrays."""
    psd, freqs = core.power_spectrum(np.ones(1, dtype=np.int8))
    assert psd.size == 0 and freqs.size == 0


def test_cv_isi_values_and_undefined_cases() -> None:
    """A regular train has CV 0; too few intervals or a zero mean give nan."""
    regular = np.zeros(100, dtype=np.int8)
    regular[::10] = 1
    assert core.cv_isi(regular) == pytest.approx(0.0)
    irregular = _train(0, 1, 11, length=20)
    assert core.cv_isi(irregular) == pytest.approx(np.std([1, 10]) / np.mean([1, 10]))
    assert math.isnan(core.cv_isi(_train(0, 5, length=10)))
    assert math.isnan(core.cv_isi(regular, dt=0.0))


def test_fano_factor_values_and_undefined_cases() -> None:
    """Window counts set the factor; short or silent trains give nan."""
    train = np.zeros(40, dtype=np.int8)
    train[[0, 1, 2, 10, 20, 21]] = 1
    counts = np.array([3, 1, 2, 0])
    assert core.fano_factor(train, window_ms=10.0) == pytest.approx(counts.var() / counts.mean())
    assert math.isnan(core.fano_factor(train[:5], window_ms=10.0))
    assert math.isnan(core.fano_factor(np.zeros(40, dtype=np.int8), window_ms=10.0))
    # A window shorter than one step still uses one step.
    assert core.fano_factor(train, window_ms=0.1) == pytest.approx(train.var() / train.mean())


@pytest.mark.parametrize(
    ("order", "delay", "seed"), list(itertools.product((2, 3, 4), (1, 2, 3), (0, 1)))
)
def test_permutation_entropy_equals_the_loop_oracle(order: int, delay: int, seed: int) -> None:
    """Binary trains, full of ties, give the oracle's value for every order and delay."""
    rng = np.random.default_rng(seed)
    train = (rng.random(400) < 0.3).astype(np.int8)
    expected = _loop_permutation_entropy(train, order, delay)
    assert core.permutation_entropy(train, order, delay) == pytest.approx(expected, rel=1e-12)


def test_permutation_entropy_reference_values() -> None:
    """A monotone series has one pattern; distinct random values approach the maximum."""
    assert core.permutation_entropy(np.arange(50.0), order=3) == 0.0
    rng = np.random.default_rng(5)
    assert core.permutation_entropy(rng.random(20000), order=3) > 0.99
    # Ties are ranked by position: an all-equal series is the identity pattern only.
    assert core.permutation_entropy(np.ones(30), order=4, delay=2) == 0.0


def test_permutation_entropy_edge_cases() -> None:
    """Too short gives nan, order 1 gives 0.0, and order or delay below 1 is refused."""
    assert math.isnan(core.permutation_entropy(np.zeros(5), order=3, delay=2))
    assert core.permutation_entropy(np.zeros(6), order=3, delay=2) == 0.0
    assert core.permutation_entropy(np.array([0, 1, 1, 0, 1]), order=1) == 0.0
    for order, delay in ((0, 1), (3, 0), (-1, 1), (3, -2)):
        with pytest.raises(ValueError, match="order and delay must be at least 1"):
            core.permutation_entropy(np.zeros(10), order=order, delay=delay)


def test_analysis_permutation_entropy_uses_the_core_without_the_rust_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the helper switched off the analysis name returns the core result."""
    monkeypatch.setattr(analysis_variability, "_HAS_RUST", False)
    train = (np.random.default_rng(3).random(300) < 0.4).astype(np.int8)
    assert analysis_variability.permutation_entropy(train, 4, 2) == core.permutation_entropy(
        train, 4, 2
    )


def test_analysis_permutation_entropy_sends_only_valid_requests_to_the_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short trains and invalid parameters reach the core, never the helper."""
    calls: list[tuple[int, int]] = []

    def helper(values: np.ndarray[Any, Any], order: int, delay: int) -> float:
        calls.append((order, delay))
        return 0.25

    monkeypatch.setattr(analysis_variability, "_HAS_RUST", True)
    monkeypatch.setattr(
        analysis_variability, "_ssc", SimpleNamespace(py_permutation_entropy=helper)
    )
    assert analysis_variability.permutation_entropy(np.zeros(9), 3, 1) == 0.25
    assert math.isnan(analysis_variability.permutation_entropy(np.zeros(5), 3, 2))
    with pytest.raises(ValueError):
        analysis_variability.permutation_entropy(np.zeros(9), 0, 1)
    with pytest.raises(ValueError):
        analysis_variability.permutation_entropy(np.zeros(9), 3, 0)
    assert calls == [(3, 1)]


@pytest.mark.skipif(
    not (analysis_variability._HAS_RUST and analysis_variability._ssc is not None),
    reason="the Rust spike_stats_core helper is not built in this checkout",
)
@pytest.mark.parametrize(("order", "delay"), [(3, 1), (4, 2), (6, 1)])
def test_permutation_entropy_matches_the_rust_helper(order: int, delay: int) -> None:
    """Where the helper is built, it and the core agree on binary trains."""
    train = (np.random.default_rng(order * 10 + delay).random(5000) < 0.1).astype(np.int8)
    helper = analysis_variability._ssc
    assert helper is not None
    rust = float(
        helper.py_permutation_entropy(np.ascontiguousarray(train.astype(np.float64)), order, delay)
    )
    assert core.permutation_entropy(train, order, delay) == pytest.approx(rust, rel=1e-12)

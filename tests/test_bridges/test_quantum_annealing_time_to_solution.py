# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing time-to-solution contracts

from __future__ import annotations

import math

import pytest

from sc_neurocore.bridges.quantum_annealing import (
    TTSAnalyzer,
)
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


def test_tts_compute_boundary_cases() -> None:
    """TTS handles zero, perfect, and interior success probabilities."""
    analyzer = TTSAnalyzer()
    impossible = analyzer.compute(0.0, 20.0)
    assert math.isinf(impossible["tts_us"])
    perfect = analyzer.compute(1.0, 20.0)
    assert perfect["n_runs_needed"] == 1.0
    interior = analyzer.compute(0.5, 20.0)
    assert interior["tts_us"] > 20.0
    assert interior["tts_ms"] == interior["tts_us"] / 1000.0


def test_tts_from_samples_and_comparison() -> None:
    """Observed energy counts feed comparable named solver rows."""
    analyzer = TTSAnalyzer()
    row = analyzer.from_samples([-2.0, -2.0, 0.0], -2.0, tolerance=1e-9)
    assert row["p_success"] == pytest.approx(2 / 3)
    empty = analyzer.from_samples([], -2.0)
    assert math.isinf(empty["tts_us"])
    comparison = analyzer.compare_solvers(
        {
            "python": {"energies": [-2.0, 0.0], "t_anneal_us": 40.0},
            "native": {"energies": [-2.0, -2.0]},
        },
        -2.0,
    )
    assert set(comparison) == {"python", "native"}
    assert comparison["native"]["p_success"] == 1.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: TTSAnalyzer().compute(float("nan"), 20.0), "finite"),
        (lambda: TTSAnalyzer().compute(-0.1, 20.0), "between"),
        (lambda: TTSAnalyzer().compute(1.1, 20.0), "between"),
        (lambda: TTSAnalyzer().compute(0.5, 20.0, 0.0), "strictly"),
        (lambda: TTSAnalyzer().compute(0.5, 20.0, 1.0), "strictly"),
        (lambda: TTSAnalyzer().compute(0.5, 0.0), "greater than zero"),
        (lambda: TTSAnalyzer().from_samples(unsafe("bad"), -1.0), "sequence"),
        (lambda: TTSAnalyzer().from_samples([0.0], float("inf")), "finite"),
        (lambda: TTSAnalyzer().from_samples([0.0], 0.0, tolerance=0.0), "tolerance"),
        (lambda: TTSAnalyzer().from_samples([float("nan")], 0.0), "finite"),
        (lambda: TTSAnalyzer().compare_solvers({"": {"energies": []}}, 0.0), "names"),
        (lambda: TTSAnalyzer().compare_solvers({"x": {"energies": "bad"}}, 0.0), "energy sequence"),
        (
            lambda: TTSAnalyzer().compare_solvers(
                {"x": {"energies": [], "t_anneal_us": True}}, 0.0
            ),
            "numeric",
        ),
    ],
)
def test_tts_rejects_invalid_inputs(call: object, match: str) -> None:
    """TTS probabilities, times, energies, and solver payloads are validated."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()

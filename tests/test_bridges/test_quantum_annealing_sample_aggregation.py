# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing sample-aggregation contracts

from __future__ import annotations


import pytest

from sc_neurocore.bridges.quantum_annealing import (
    SampleAggregator,
)
from tests.test_bridges.quantum_annealing_test_helpers import unsafe


def test_sample_aggregator_statistics() -> None:
    """Aggregation deduplicates, bins, and Boltzmann-weights aligned samples."""
    samples = [{0: 1, 1: -1}, {0: 1, 1: -1}, {0: -1, 1: 1}]
    result = SampleAggregator().aggregate(samples, [-2.0, -2.0, 0.0], temperature=0.5)
    assert result["unique_samples"] == 2
    assert result["best_sample"] == samples[0]
    assert result["best_energy"] == -2.0
    assert result["success_probability"] == pytest.approx(2 / 3)
    assert result["gs_degeneracy"] == 2
    assert len(result["histogram"]["counts"]) == 2
    assert result["boltzmann_avg_energy"] < result["mean_energy"]
    assert SampleAggregator().aggregate([], []) == {
        "unique_samples": 0,
        "best": {},
        "histogram": {},
    }


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SampleAggregator().aggregate(unsafe("bad"), []), "sequences"),
        (lambda: SampleAggregator().aggregate([{}], []), "equal lengths"),
        (lambda: SampleAggregator().aggregate([{}], [0.0], 0.0), "greater than zero"),
        (lambda: SampleAggregator().aggregate([{unsafe(-1): 1}], [0.0]), "indices"),
        (lambda: SampleAggregator().aggregate([{0: 2}], [0.0]), "domain"),
        (lambda: SampleAggregator().aggregate([{}], [float("nan")]), "finite"),
    ],
)
def test_sample_aggregator_rejects_invalid_inputs(call: object, match: str) -> None:
    """Misaligned, malformed, and non-finite sample sets are rejected."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()

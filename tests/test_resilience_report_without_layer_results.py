# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a resilience report whose experiments name no layer

"""A report with only whole-network experiments names no vulnerable layer and says so in its summary."""

from __future__ import annotations

from sc_neurocore.resilience.fault_suite import FaultResult, FaultType, ResilienceReport


def _whole_network_result(rate: float, degradation: float) -> FaultResult:
    """Return one result of a fault applied to every layer at once."""
    return FaultResult(
        fault_type=FaultType.DEAD_SYNAPSE,
        fault_rate=rate,
        layer_index=None,
        accuracy_before=0.9,
        accuracy_after=0.9 - degradation,
        degradation=degradation,
    )


def test_no_layer_is_named_when_no_experiment_targets_a_layer() -> None:
    """Without a per-layer experiment there is no layer to rank, so none is returned or reported."""
    report = ResilienceReport(
        results=[_whole_network_result(0.1, 0.05), _whole_network_result(0.2, 0.2)]
    )
    assert report.most_vulnerable_layer() is None
    summary = report.summary()
    assert summary.splitlines()[0] == "Fault Resilience Report: 2 experiments"
    assert "dead_synapse: mean_deg=0.125, max_deg=0.200" in summary
    assert "Most vulnerable layer" not in summary


def test_an_empty_report_names_no_layer() -> None:
    """A report with no experiments has no layer to rank."""
    report = ResilienceReport()
    assert report.most_vulnerable_layer() is None
    assert report.summary() == "Fault Resilience Report: 0 experiments"

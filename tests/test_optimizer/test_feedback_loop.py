# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optimiser synthesis feedback loop tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.optimizer import (
    ObservationLoadError,
    optimise_from_evidence_payload,
    optimise_from_synthesis_reports,
)
from sc_neurocore.optimizer.feedback_loop import SynthesisFeedbackResult
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import TargetHardwareProfile


def _target() -> TargetHardwareProfile:
    return TargetHardwareProfile(
        name="feedback-fpga",
        budget=HardwareBudget(max_luts=10_000, max_power_mw=100.0, max_latency_cycles=512),
    )


def _network() -> list[LayerProfile]:
    return [LayerProfile(id="dense0", mac_count=128, is_critical_path=True)]


def _write_design(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "mac_count": 128,
                "bitstream_length": 64,
                "decorrelator": "SCC_Decorrelator",
                "mode": "SC",
                "precision_bits": 8,
                "lfsr_polynomial": "none",
                "is_critical_path": True,
            }
        ),
        encoding="utf-8",
    )


def test_synthesis_feedback_loop_parses_reports_and_updates_plan(tmp_path: Path) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 320\nLatency: 64 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 7.5\n", encoding="utf-8")

    result = optimise_from_synthesis_reports(
        network=_network(),
        target=_target(),
        design_path=design,
        utilisation_path=utilisation,
        power_path=power,
        accuracy_score=0.998,
        clock_mhz=100.0,
        inferences_per_run=4,
    )

    assert isinstance(result, SynthesisFeedbackResult)
    assert result.observations[0].luts_used == 320
    assert result.evidence_payload["energy"]["energy_uj_per_inference"] > 0.0
    assert result.report.feasible
    assert result.report.training_points > len(result.observations)
    chosen = result.report.config["dense0"]
    assert chosen.mode == "SC"
    assert chosen.luts_used <= result.report.total_luts
    assert chosen.accuracy_score >= 0.9


def test_synthesis_feedback_loop_without_energy_metadata_omits_energy_payload(
    tmp_path: Path,
) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 300\nLatency: 80 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 8.0\n", encoding="utf-8")

    result = optimise_from_synthesis_reports(
        network=_network(),
        target=_target(),
        design_path=design,
        utilisation_path=utilisation,
        power_path=power,
        accuracy_score=0.991,
    )
    assert "energy" not in result.evidence_payload
    assert result.observations[0].latency_cycles == 80


def test_synthesis_feedback_loop_latency_cycles_override_is_honoured(
    tmp_path: Path,
) -> None:
    design = tmp_path / "design.json"
    utilisation = tmp_path / "utilisation.rpt"
    power = tmp_path / "power.rpt"
    _write_design(design)
    utilisation.write_text("CLB LUTs | 300\nLatency: 80 cycles\n", encoding="utf-8")
    power.write_text("Total On-Chip Power (mW): 8.0\n", encoding="utf-8")

    result = optimise_from_synthesis_reports(
        network=_network(),
        target=_target(),
        design_path=design,
        utilisation_path=utilisation,
        power_path=power,
        accuracy_score=0.991,
        latency_cycles=55,
    )
    assert result.observations[0].latency_cycles == 55


def test_feedback_payload_rejects_incomplete_evidence() -> None:
    with pytest.raises(ObservationLoadError, match="missing"):
        optimise_from_evidence_payload(
            network=_network(),
            target=_target(),
            payload={"observations": [{"mac_count": 128}]},
        )


def test_feedback_loop_fails_closed_when_surrogate_returns_no_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sc_neurocore.optimizer.feedback_loop as feedback_loop_mod

    class _NullOptimiser:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def optimise(self, network: object) -> None:
            return None

    monkeypatch.setattr(feedback_loop_mod, "SurrogateSCOptimizer", _NullOptimiser)
    with pytest.raises(RuntimeError, match="returned no report"):
        optimise_from_evidence_payload(
            network=_network(),
            target=_target(),
            payload={
                "observations": [
                    {
                        "mac_count": 128,
                        "bitstream_length": 64,
                        "decorrelator": "SCC_Decorrelator",
                        "mode": "SC",
                        "precision_bits": 8,
                        "lfsr_polynomial": "none",
                        "luts_used": 320,
                        "power_mw": 7.5,
                        "latency_cycles": 64,
                        "accuracy_score": 0.99,
                        "is_critical_path": True,
                    }
                ]
            },
        )

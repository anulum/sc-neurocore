# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Focused protocol, ensemble, and source-custody tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from sc_neurocore.network import (
    SC_COMPTE_WM_BEHAVIOR_BACKENDS,
    SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS,
    SCCompteWMBehaviorProtocol,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BEHAVIOR = REPOSITORY / "src/sc_neurocore/network/sc_compte_wm_behavior.py"
BENCHMARK = REPOSITORY / "benchmarks/bench_sc_compte_wm_behavior_ensemble.py"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_behavior_ensemble.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_behavior_protocol_freezes_named_non_overlapping_epochs() -> None:
    protocol = SCCompteWMBehaviorProtocol()
    assert protocol.duration_ms == 2500.0
    assert protocol.duration_ms / protocol.window_ms == 10
    cue, distractor, response = protocol.stimuli()
    assert (cue.start_ms, cue.center_deg, cue.current_pa) == (250.0, 180.0, 200.0)
    assert (distractor.start_ms, distractor.center_deg, distractor.current_pa) == (
        1000.0,
        270.0,
        200.0,
    )
    assert response.kind == "global_current" and response.center_deg is None
    assert response.start_ms == 1750.0 and response.current_pa == 500.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"window_ms": 200.0}, "ten statistics windows"),
        ({"response_start_ms": 2400.0}, "inside the run"),
        ({"cue_current_pa": 0.0}, "must be positive"),
    ],
)
def test_behavior_protocol_rejects_ambiguous_or_out_of_run_variants(
    change: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(SCCompteWMBehaviorProtocol(), **change)


def test_committed_behavior_ensemble_closes_every_predeclared_check() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.sc-compte-wm-behavior-ensemble.v1"
    assert payload["model"] == "SC-COMPTE-WM-NETWORK"
    assert payload["evidence_class"] == "deterministic_simulator_behavior_ensemble"
    assert payload["source_reproduction_claimed"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["production_speed_claimed"] is False
    assert payload["persistent_bump_claimed"] is True
    assert payload["distractor_resistance_claimed"] is True
    assert payload["response_reset_claimed"] is True
    assert payload["reference"] == {
        "backend": "rust",
        "seeds": list(SC_COMPTE_WM_BEHAVIOR_REFERENCE_SEEDS),
    }
    assert payload["ensemble"]["represented_backends"] == list(SC_COMPTE_WM_BEHAVIOR_BACKENDS)
    assert payload["ensemble"]["anchor_seed"] == 42
    assert payload["ensemble"]["all_runtime_input_spike_count_exact"] is True
    assert payload["ensemble"]["bidirectional_seed_drift"] is True
    assert payload["ensemble"]["all_trials_passed"] is True
    assert payload["passed"] is True
    assert len(payload["trials"]) == 7
    assert all(trial["passed"] and all(trial["checks"].values()) for trial in payload["trials"])


def test_behavior_receipt_is_bound_to_protocol_and_all_runtime_sources() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    required = (
        BEHAVIOR,
        BENCHMARK,
        REPOSITORY / "src/sc_neurocore/network/sc_compte_wm_backends.py",
        REPOSITORY / "engine/src/sc_compte_wm_network.rs",
        REPOSITORY / "src/sc_neurocore/accel/julia/sc_compte_wm_network/SCCompteWMNetwork.jl",
        REPOSITORY / "src/sc_neurocore/accel/go/sc_compte_wm_network/network.go",
    )
    for path in required:
        relative = path.relative_to(REPOSITORY).as_posix()
        assert payload["source_sha256"][relative] == _sha256(path)


def test_behavior_public_api_is_documented_and_explicitly_scoped() -> None:
    source = BEHAVIOR.read_text(encoding="utf-8")
    assert "Ensemble behavior contract" in source
    assert "without\naltering or relabelling" in source
    for symbol in (
        "SCCompteWMBehaviorProtocol",
        "SCCompteWMBehaviorAcceptance",
        "SCCompteWMBehaviorMetrics",
        "SCCompteWMBehaviorTrial",
        "SCCompteWMBehaviorEnsemble",
        "assess_sc_compte_wm_behavior",
        "run_sc_compte_wm_behavior_trial",
        "summarize_sc_compte_wm_behavior_ensemble",
    ):
        assert f'"{symbol}"' in source

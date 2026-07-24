# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (formal_report_validation) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403


def test_validate_formal_network_report_accepts_complete_payload() -> None:
    payload = _valid_formal_report_payload()

    validate_formal_network_report(payload)


def test_validate_formal_network_report_rejects_symlink_artifact_path(tmp_path: Path) -> None:
    payload = _valid_formal_report_payload()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    _materialise_formal_report_artifacts(payload, artifact_root)

    target = artifact_root / "dense_lif_frontier_fixture_rate_bound.sv"
    symlink = artifact_root / "symlink_rate_bound.sv"
    symlink.symlink_to(target)
    payload["artifacts"]["sva"] = str(symlink)
    payload["artifacts"]["rate_sva"] = str(symlink)

    with pytest.raises(FormalReportValidationError, match="must not be a symlink"):
        validate_formal_network_report(payload, artifact_root=artifact_root)


def test_validate_formal_network_report_rejects_directory_artifact_path(tmp_path: Path) -> None:
    payload = _valid_formal_report_payload()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    _materialise_formal_report_artifacts(payload, artifact_root)

    directory = artifact_root / "fake_dir.sv"
    directory.mkdir()
    payload["artifacts"]["formal_bundle"] = str(directory)

    with pytest.raises(FormalReportValidationError, match="must be a regular file"):
        validate_formal_network_report(payload, artifact_root=artifact_root)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "schema_version"),
        (lambda payload: payload["artifacts"].pop("rtl"), "artifacts.rtl"),
        (lambda payload: payload["network"].__setitem__("output_width", 0), "output_width"),
        (lambda payload: payload["rate_bound"].__setitem__("max_spikes", 9), "max_spikes"),
        (
            lambda payload: payload["symbiyosys"].__setitem__("status", "unknown"),
            "symbiyosys.status",
        ),
        (lambda payload: payload.__setitem__("rate_replay", {"violated": False}), "rate_replay"),
        (
            lambda payload: payload["temporal_replay"].__setitem__("trigger_output", 9),
            "temporal_replay.trigger_output",
        ),
        (
            lambda payload: payload["temporal_replay"].__setitem__("violating_output", 9),
            "temporal_replay.violating_output",
        ),
        (
            lambda payload: (
                payload["temporal_replay"].__setitem__("trigger_output", 0),
                payload["temporal_replay"].__setitem__("violating_output", 0),
            ),
            "temporal_replay.violating_output",
        ),
        (
            lambda payload: payload["population_coactivation"].__setitem__("max_active_outputs", 3),
            "population_coactivation.max_active_outputs",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__("max_active_outputs", 2),
            "population_replay.max_active_outputs",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__("observed_active_outputs", 3),
            "population_replay.observed_active_outputs",
        ),
        (
            lambda payload: (
                payload["population_replay"].__setitem__("violated", True),
                payload["population_replay"].__setitem__("first_violation_cycle", None),
                payload["population_replay"].__setitem__("observed_active_outputs", 2),
            ),
            "population_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__("observed_active_outputs", 2),
            "population_replay.observed_active_outputs",
        ),
        (
            lambda payload: payload["population_silence"].__setitem__("trigger_active_outputs", 3),
            "population_silence.trigger_active_outputs",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "trigger_active_outputs", 3
            ),
            "population_silence_replay.trigger_active_outputs",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", None),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", None),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 2),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
            ),
            "population_silence_replay.trigger_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 2),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 2),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
            ),
            "population_silence_replay.trigger_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 4),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "remaining_silence_cycles", 3
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 2),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
                payload["population_silence_replay"].__setitem__("remaining_silence_cycles", 2),
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("cycles_checked", 2),
                payload["population_silence_replay"].__setitem__("remaining_silence_cycles", 2),
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "remaining_silence_cycles", 1
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 3),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 1),
                payload["population_silence_replay"].__setitem__("cycles_checked", 5),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "observed_active_outputs", 1
            ),
            "population_silence_replay.observed_active_outputs",
        ),
        (
            lambda payload: payload["population_inactivity"].__setitem__("max_silent_cycles", 0),
            "population_inactivity.max_silent_cycles",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "max_silent_cycles", 3
            ),
            "population_inactivity_replay.max_silent_cycles",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__("first_violation_cycle", None),
                payload["population_inactivity_replay"].__setitem__("observed_silent_cycles", 3),
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__("first_violation_cycle", 4),
                payload["population_inactivity_replay"].__setitem__("observed_silent_cycles", 3),
                payload["population_inactivity_replay"].__setitem__("cycles_checked", 4),
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__("first_violation_cycle", 3),
                payload["population_inactivity_replay"].__setitem__("observed_silent_cycles", 4),
            ),
            "population_inactivity_replay.observed_silent_cycles",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "first_violation_cycle", 3
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "observed_silent_cycles", 3
            ),
            "population_inactivity_replay.observed_silent_cycles",
        ),
        # --- network/rate-bound and sub-spec construction failures ---
        (lambda payload: payload.__setitem__("network", 5), "network must be an object"),
        (
            lambda payload: payload["rate_bound"].__setitem__("output_index", 5),
            "rate_bound.output_index must exist",
        ),
        (
            lambda payload: payload["refractory"].__setitem__("refractory_cycles", 0),
            "refractory_cycles must be a positive integer",
        ),
        (
            lambda payload: payload["refractory"].__setitem__("output_index", 5),
            "refractory.output_index must exist",
        ),
        (
            lambda payload: payload["antagonistic_exclusion"].__setitem__("output_a", -1),
            "output_a must be a non-negative",
        ),
        (
            lambda payload: payload["antagonistic_exclusion"].__setitem__("output_a", 5),
            "antagonistic_exclusion.output_a must exist",
        ),
        (
            lambda payload: payload["antagonistic_exclusion"].__setitem__("output_b", 5),
            "antagonistic_exclusion.output_b must exist",
        ),
        (
            lambda payload: payload["temporal_separation"].__setitem__("separation_cycles", 0),
            "separation_cycles must be a positive integer",
        ),
        (
            lambda payload: payload["temporal_separation"].__setitem__("output_a", 5),
            "temporal_separation.output_a must exist",
        ),
        (
            lambda payload: payload["temporal_separation"].__setitem__("output_b", 5),
            "temporal_separation.output_b must exist",
        ),
        (
            lambda payload: payload["population_coactivation"].__setitem__(
                "max_active_outputs", -1
            ),
            "max_active_outputs must be a non-negative",
        ),
        (
            lambda payload: payload["population_silence"].__setitem__("silence_cycles", 0),
            "silence_cycles must be a positive integer",
        ),
        # --- artifacts.* must-be-null when the matching property is absent ---
        (
            lambda payload: payload.__setitem__("refractory", None),
            "artifacts.refractory_sva must be null",
        ),
        (
            lambda payload: payload.__setitem__("antagonistic_exclusion", None),
            "artifacts.antagonistic_sva must be null",
        ),
        (
            lambda payload: payload.__setitem__("temporal_separation", None),
            "artifacts.temporal_sva must be null",
        ),
        (
            lambda payload: payload.__setitem__("population_coactivation", None),
            "artifacts.population_sva must be null",
        ),
        (
            lambda payload: payload.__setitem__("population_silence", None),
            "artifacts.population_silence_sva must be null",
        ),
        (
            lambda payload: payload.__setitem__("population_inactivity", None),
            "artifacts.population_inactivity_sva must be null",
        ),
        (
            lambda payload: payload["artifacts"].__setitem__("sva", "/tmp/mismatched.sv"),
            "artifacts.sva must match artifacts.rate_sva",
        ),
        # --- replay must-be-null when the matching property is absent ---
        (
            lambda payload: (
                payload.__setitem__("refractory", None),
                payload["artifacts"].__setitem__("refractory_sva", None),
            ),
            "refractory_replay must be null",
        ),
        (
            lambda payload: (
                payload.__setitem__("antagonistic_exclusion", None),
                payload["artifacts"].__setitem__("antagonistic_sva", None),
            ),
            "antagonistic_replay must be null",
        ),
        (
            lambda payload: (
                payload.__setitem__("temporal_separation", None),
                payload["artifacts"].__setitem__("temporal_sva", None),
            ),
            "temporal_replay must be null",
        ),
        (
            lambda payload: (
                payload.__setitem__("population_coactivation", None),
                payload["artifacts"].__setitem__("population_sva", None),
            ),
            "population_replay must be null",
        ),
        (
            lambda payload: (
                payload.__setitem__("population_silence", None),
                payload["artifacts"].__setitem__("population_silence_sva", None),
            ),
            "population_silence_replay must be null",
        ),
        (
            lambda payload: (
                payload.__setitem__("population_inactivity", None),
                payload["artifacts"].__setitem__("population_inactivity_sva", None),
            ),
            "population_inactivity_replay must be null",
        ),
        (
            lambda payload: payload["replay"].__setitem__("observed_spikes", 99),
            "replay must match rate_replay",
        ),
        # --- symbiyosys metadata invariants ---
        (
            lambda payload: payload["symbiyosys"].__setitem__("requested", 1),
            "symbiyosys.requested must be a boolean",
        ),
        (
            lambda payload: payload["symbiyosys"].__setitem__("returncode", "x"),
            "symbiyosys.returncode must be int or null",
        ),
        (
            lambda payload: payload["symbiyosys"].__setitem__("stdout", 5),
            "symbiyosys.stdout must be a string",
        ),
        (
            lambda payload: payload["symbiyosys"].__setitem__("sby", "/tmp/other.sby"),
            "symbiyosys.sby must match artifacts.sby",
        ),
        (
            lambda payload: payload["rate_replay"].__setitem__("violated", 5),
            "rate_replay.violated must be a boolean",
        ),
        # --- antagonistic replay output binding ---
        (
            lambda payload: payload["antagonistic_replay"].__setitem__("output_a", 1),
            "antagonistic_replay.output_a must match",
        ),
        (
            lambda payload: payload["antagonistic_replay"].__setitem__("output_b", 0),
            "antagonistic_replay.output_b must match",
        ),
        # --- population coactivation replay timing ---
        (
            lambda payload: (
                payload["population_replay"].__setitem__("violated", True),
                payload["population_replay"].__setitem__("first_violation_cycle", 2),
                payload["population_replay"].__setitem__("observed_active_outputs", 1),
            ),
            "observed_active_outputs must exceed max_active_outputs when violated",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__("first_violation_cycle", 2),
            "population_replay.first_violation_cycle must be null when not violated",
        ),
        # --- population silence replay timing ---
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "observed_active_outputs", 5
            ),
            "population_silence_replay.observed_active_outputs must be <= network output_width",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__("silence_cycles", 5),
            "population_silence_replay.silence_cycles must match",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("first_violation_cycle", 2),
                payload["population_silence_replay"].__setitem__("observed_active_outputs", 0),
            ),
            "population_silence_replay.observed_active_outputs must be positive when violated",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "first_violation_cycle", 2
            ),
            "population_silence_replay.first_violation_cycle must be null when not violated",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__("trigger_cycle", 4),
            "population_silence_replay.trigger_cycle must be less than cycles_checked",
        ),
        # --- population inactivity replay timing ---
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__("first_violation_cycle", 3),
                payload["population_inactivity_replay"].__setitem__("observed_silent_cycles", 2),
            ),
            "observed_silent_cycles must exceed max_silent_cycles when violated",
        ),
    ],
)
def test_validate_formal_network_report_rejects_invalid_payloads(mutator, match: str) -> None:
    payload = _valid_formal_report_payload()
    mutator(payload)

    with pytest.raises(FormalReportValidationError, match=match):
        validate_formal_network_report(payload)


@pytest.mark.parametrize(
    "nuller",
    [
        lambda payload: (
            payload.__setitem__("rate_replay", None),
            payload.__setitem__("replay", None),
        ),
        lambda payload: payload.__setitem__("refractory_replay", None),
        lambda payload: payload.__setitem__("antagonistic_replay", None),
        lambda payload: payload.__setitem__("temporal_replay", None),
        lambda payload: payload.__setitem__("population_replay", None),
        lambda payload: payload.__setitem__("population_silence_replay", None),
        lambda payload: payload.__setitem__("population_inactivity_replay", None),
    ],
)
def test_validate_formal_network_report_accepts_null_replays(nuller) -> None:
    """A present property with a null replay record is accepted (replay is optional)."""
    payload = _valid_formal_report_payload()
    nuller(payload)

    validate_formal_network_report(payload)


def test_validate_formal_network_report_rejects_missing_artifact_file(tmp_path: Path) -> None:
    """An artifact path under the root that does not exist is rejected."""
    payload = _valid_formal_report_payload()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    for key, raw_path in artifacts.items():
        if raw_path is not None:
            artifacts[key] = str(tmp_path / Path(str(raw_path)).name)

    with pytest.raises(FormalReportValidationError, match="does not exist"):
        validate_formal_network_report(payload, artifact_root=tmp_path)


def test_validate_formal_network_report_rejects_artifact_outside_root(tmp_path: Path) -> None:
    """A materialised artifact located outside the artifact root is rejected."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    payload = _valid_formal_report_payload()
    _materialise_formal_report_artifacts(payload, outside)

    with pytest.raises(FormalReportValidationError, match="is outside artifact_root"):
        validate_formal_network_report(payload, artifact_root=root)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio replay pack contract

"""The replay pack must reproduce an experiment, and refuse when it cannot.

The cases here are the ones that used to pass silently: a request whose time
step the export dropped, a drive protocol it never carried, a model whose step
signature it guessed wrong, and a stochastic run whose seed it never pinned.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.experiment_spec import resolve_experiment, run_experiment
from sc_neurocore.studio.replay_pack import (
    REPLAY_PACK_SCHEMA_VERSION,
    ReplayRejected,
    build_replay_pack,
    compare_to_expectation,
    experiment_identity,
    experiment_identity_sha256,
    load_replay_pack,
    main,
    pinned_request,
    replay_expectation,
    replay_pack,
    verify_replay_pack,
)

# A model whose public step takes ``drive``, not ``current``: the old export
# wrote ``neuron.step(current=…)`` and this model's script died on it.
LAPICQUE = {
    "name": "SCLapicqueLIFNeuron",
    "dt": 0.25,
    "duration": 60.0,
    "current": 1.5,
    "protocol": "ramp",
}
# Nondefault dt (the model's own default is 0.01) and a non-constant drive:
# the old export ran the model at its default step and a constant current.
HODGKIN_HUXLEY = {
    "name": "HodgkinHuxleyNeuron",
    "dt": 0.05,
    "duration": 50.0,
    "current": 10.0,
    "protocol": "step",
}
STOCHASTIC_EQUATIONS = {
    "equations": ["dv/dt = (-(v) + I)/tau + 0.4*xi"],
    "threshold": "v > 1",
    "reset": "v = 0",
    "params": {"tau": 10.0},
    "init": {"v": 0.0},
    "dt": 0.1,
    "duration": 100.0,
    "current": 1.2,
}


def _run(request: dict[str, Any]) -> dict[str, Any]:
    return run_experiment(resolve_experiment(request))


class TestPinning:
    @pytest.mark.parametrize("params", [{"seed": 77}, {"seed": 77, "rate_hz": 150}])
    def test_model_parameter_seed_is_pinned_without_duplicate_declaration(
        self, params: dict[str, int]
    ) -> None:
        """Seeded catalogue replay retains other parameters but declares its seed once."""
        pack = build_replay_pack({"name": "PoissonNeuron", "params": params, "duration": 10.0})
        assert pack["request"]["seed"] == 77
        assert "seed" not in pack["request"].get("params", {})
        if "rate_hz" in params:
            assert pack["request"]["params"]["rate_hz"] == 150
        assert replay_pack(pack)["verdict"] == "match"

    def test_a_fresh_trial_is_sealed_as_the_replay_of_its_drawn_seed(self) -> None:
        request = dict(STOCHASTIC_EQUATIONS, trial="fresh")
        spec = resolve_experiment(request)
        drawn = spec.public["randomness"]["seed"]
        assert spec.public["randomness"]["effective_trial"] == "fresh"

        sealed = pinned_request(request, spec)

        assert sealed["seed"] == drawn
        assert sealed["trial"] == "replay"
        replayed = resolve_experiment(sealed)
        assert replayed.public["randomness"]["seed"] == drawn
        assert replayed.public["randomness"]["effective_trial"] == "replay"
        assert replayed.cacheable is True

    def test_a_deterministic_experiment_is_not_given_a_seed(self) -> None:
        spec = resolve_experiment(HODGKIN_HUXLEY)
        sealed = pinned_request(HODGKIN_HUXLEY, spec)
        assert "seed" not in sealed
        # The contract refuses a seed here, so pinning one would break replay.
        resolve_experiment(sealed)

    def test_unknown_request_keys_never_enter_the_pack(self) -> None:
        spec = resolve_experiment(HODGKIN_HUXLEY)
        sealed = pinned_request({**HODGKIN_HUXLEY, "mode": "model", "nonsense": 1}, spec)
        assert "mode" not in sealed
        assert "nonsense" not in sealed


class TestIdentity:
    def test_identity_excludes_the_runtime_and_the_cache_key(self) -> None:
        public = resolve_experiment(HODGKIN_HUXLEY).public
        identity = experiment_identity(public)
        assert "runtime" not in identity
        assert "cache" not in identity
        assert "experiment_sha256" not in identity
        assert identity["model"] == public["model"]
        assert identity["protocol"] == public["protocol"]

    def test_a_different_timestep_is_a_different_identity(self) -> None:
        one = experiment_identity_sha256(resolve_experiment(HODGKIN_HUXLEY).public)
        other = experiment_identity_sha256(
            resolve_experiment({**HODGKIN_HUXLEY, "dt": 0.02}).public
        )
        assert one != other

    def test_a_different_protocol_is_a_different_identity(self) -> None:
        one = experiment_identity_sha256(resolve_experiment(HODGKIN_HUXLEY).public)
        other = experiment_identity_sha256(
            resolve_experiment({**HODGKIN_HUXLEY, "protocol": "constant"}).public
        )
        assert one != other


class TestBuildAndReplay:
    def test_vector_trajectory_and_snapshots_replay_in_full(self) -> None:
        """The real neural-field result carries every vector component at each step."""
        pack = build_replay_pack(
            {"name": "AmariNeuralField", "duration": 2.0, "dt": 0.1, "current": 1.0}
        )
        assert len(pack["expectation"]["vector_states"]["u"]["samples"]) == 20
        assert replay_pack(pack)["verdict"] == "match"
        changed = _run(pack["request"])
        changed["raw"]["vector_states"]["u"][10][3] += 0.1
        outcome = compare_to_expectation(pack["expectation"], changed, tolerance=1e-6)
        assert outcome["verdict"] == "mismatch"
        assert outcome["worst_state_deviation"] == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "request_body",
        [LAPICQUE, HODGKIN_HUXLEY, dict(STOCHASTIC_EQUATIONS, seed=4242)],
        ids=["step-signature", "nondefault-dt-and-protocol", "seeded-stochastic"],
    )
    def test_a_pack_replays_the_experiment_it_sealed(self, request_body: dict[str, Any]) -> None:
        pack = build_replay_pack(request_body)
        assert pack["schema_version"] == REPLAY_PACK_SCHEMA_VERSION

        outcome = replay_pack(pack)

        assert outcome["verdict"] == "match"
        assert outcome["differences"] == []
        assert outcome["experiment_identity_sha256"] == pack["experiment_identity_sha256"]

    def test_the_pack_records_the_experiment_the_studio_would_run(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        direct = _run(HODGKIN_HUXLEY)

        assert pack["expectation"]["n_steps"] == direct["n_steps"] == 1000
        assert pack["expectation"]["dt"] == direct["dt"] == 0.05
        assert pack["expectation"]["spikes"] == list(direct["spikes"])
        assert pack["experiment"]["protocol"]["kind"] == "step"

    def test_the_expectation_covers_every_state_not_only_the_spike_count(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        states = pack["expectation"]["states"]
        # Hodgkin-Huxley carries the gating variables, not just a voltage.
        assert set(states) >= {"v", "m", "h", "n"}
        for block in states.values():
            assert block["n_samples"] == 1000
            assert block["sha256"]
        assert pack["expectation"]["trace_source"] == "raw"

    def test_a_seeded_trial_replays_to_the_same_spike_train(self) -> None:
        pack = build_replay_pack(dict(STOCHASTIC_EQUATIONS, seed=4242))
        assert pack["request"]["seed"] == 4242
        assert pack["expectation"]["spike_count"] > 0
        assert replay_pack(pack)["verdict"] == "match"

    def test_a_different_seed_produces_a_different_experiment(self) -> None:
        one = build_replay_pack(dict(STOCHASTIC_EQUATIONS, seed=1))
        other = build_replay_pack(dict(STOCHASTIC_EQUATIONS, seed=2))
        assert one["experiment_identity_sha256"] != other["experiment_identity_sha256"]
        assert one["expectation"]["spikes"] != other["expectation"]["spikes"]


class TestComparisonCatchesRealDivergence:
    """Controlled mutations of the sealed expectation, not of the run."""

    def test_a_changed_spike_train_is_a_mismatch(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["spikes"] = [index + 1 for index in mutated["spikes"]]
        mutated["spikes_sha256"] = replay_expectation(
            {**result, "spikes": mutated["spikes"], "spike_count": len(mutated["spikes"])}
        )["spikes_sha256"]

        outcome = compare_to_expectation(mutated, result)

        assert outcome["verdict"] == "mismatch"
        assert any("spike events diverge" in difference for difference in outcome["differences"])

    def test_a_changed_final_state_is_a_mismatch(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["final_state"]["v"] = float(mutated["final_state"]["v"]) + 1.0

        outcome = compare_to_expectation(mutated, result)

        assert outcome["verdict"] == "mismatch"
        assert any("final_state.v" in difference for difference in outcome["differences"])

    def test_a_changed_drive_is_a_mismatch(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["drive_sha256"] = "0" * 64

        outcome = compare_to_expectation(mutated, result)

        assert outcome["verdict"] == "mismatch"
        assert "the drive samples differ from the sealed protocol" in outcome["differences"]

    def test_a_state_within_tolerance_is_not_called_exact(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        changed = copy.deepcopy(result)
        changed["raw"]["states"]["v"][10] += 1e-9
        mutated = replay_expectation(changed)

        outcome = compare_to_expectation(mutated, result, tolerance=1e-6)

        assert outcome["verdict"] == "match-within-tolerance"
        assert outcome["worst_state_deviation"] <= 1e-6
        assert compare_to_expectation(mutated, result, tolerance=0.0)["verdict"] == "mismatch"

    @pytest.mark.parametrize("tolerance", [0.0, 1e-12])
    def test_interior_permutation_is_not_hidden_by_equal_extrema(self, tolerance: float) -> None:
        """Pointwise divergence must fail even when every stored summary agrees."""
        result = _run(dict(HODGKIN_HUXLEY, duration=5.0))
        expected = replay_expectation(result)
        changed = copy.deepcopy(result)
        trace = changed["raw"]["states"]["v"]
        trace[10], trace[11] = trace[11], trace[10]
        outcome = compare_to_expectation(expected, changed, tolerance=tolerance)
        assert outcome["verdict"] == "mismatch"
        assert outcome["worst_state_deviation"] == abs(trace[10] - trace[11])

    def test_legacy_trace_digests_cannot_prove_a_tolerance_bound(self) -> None:
        """Old packs remain exact-only when they lack full sealed samples."""
        result = _run(LAPICQUE)
        expected = replay_expectation(result)
        for state in expected["states"].values():
            del state["samples"]
        assert compare_to_expectation(expected, result)["verdict"] == "match"
        changed = copy.deepcopy(result)
        changed["raw"]["states"]["v"][10] += 1e-9
        assert compare_to_expectation(expected, changed, tolerance=1)["verdict"] == "mismatch"

    @pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
    def test_invalid_tolerance_is_refused(self, tolerance: float) -> None:
        """A nonfinite or negative threshold cannot turn divergence into success."""
        result = _run(LAPICQUE)
        with pytest.raises(ReplayRejected, match="finite and non-negative"):
            compare_to_expectation(replay_expectation(result), result, tolerance=tolerance)


def _through_a_browser(value: Any) -> Any:
    """Narrow integral floats the way ``JSON.stringify`` does.

    A pack that the browser saved and a person handed to the replay runner has
    been through JavaScript's single number type. The identity must survive it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _through_a_browser(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_through_a_browser(item) for item in value]
    return value


class TestSurvivesTheBrowser:
    def test_a_pack_saved_by_a_browser_still_replays(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        saved = json.loads(json.dumps(_through_a_browser(pack)))

        # The narrowing really happened, so the test is not vacuous.
        assert isinstance(pack["expectation"]["dt"], float)
        assert saved["experiment"]["steps"]["duration_requested_ms"] == 50
        assert isinstance(saved["experiment"]["steps"]["duration_requested_ms"], int)

        assert replay_pack(saved)["verdict"] == "match"

    def test_the_identity_digest_describes_the_json_value_not_the_python_type(self) -> None:
        public = resolve_experiment(HODGKIN_HUXLEY).public
        assert experiment_identity_sha256(public) == experiment_identity_sha256(
            _through_a_browser(public)
        )


class TestRefusalsBeforeExecution:
    @pytest.mark.parametrize(
        "field,value", [("trace_source", "display"), ("states", None), ("vector_states", [])]
    )
    def test_invalid_v2_evidence_structure_is_refused(self, field: str, value: object) -> None:
        """The versioned admission boundary rejects absent or downgraded trace groups."""
        pack = build_replay_pack(LAPICQUE)
        pack["expectation"][field] = value
        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)
        assert refusal.value.stage == "schema"

    def test_nonnumeric_samples_are_a_structured_refusal(self) -> None:
        """Malformed JSON sample values must not escape as an array conversion error."""
        pack = build_replay_pack(LAPICQUE)
        pack["expectation"]["states"]["v"]["samples"] = ["invalid"]
        with pytest.raises(ReplayRejected, match="invalid sample evidence"):
            verify_replay_pack(pack)

    def test_missing_identity_digest_is_refused(self) -> None:
        """A pack needs its scientific identity independently of valid sample hashes."""
        pack = build_replay_pack(LAPICQUE)
        del pack["experiment_identity_sha256"]
        with pytest.raises(ReplayRejected, match="experiment_identity_sha256"):
            verify_replay_pack(pack)

    @pytest.mark.parametrize("equations", [False, True])
    def test_missing_request_source_is_refused(self, equations: bool) -> None:
        """Neither a catalogue name nor an equation program may disappear from replay."""
        pack = build_replay_pack(dict(STOCHASTIC_EQUATIONS, seed=42) if equations else LAPICQUE)
        del pack["request"]["equations" if equations else "name"]
        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)
        assert refusal.value.stage == "request"

    def test_newly_invalid_seed_is_refused_by_resolution(self) -> None:
        """A deterministic model cannot acquire an ignored stochastic seed in transit."""
        pack = build_replay_pack(LAPICQUE)
        pack["request"]["seed"] = 42
        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)
        assert refusal.value.stage == "identity"
        assert "seed" in refusal.value.differences

    def test_boolean_cannot_replace_a_numeric_identity_field(self) -> None:
        """JSON booleans and numbers retain distinct identity despite Python equality."""
        pack = build_replay_pack(LAPICQUE)
        pack["experiment"]["numerical"]["substeps"] = True
        pack["experiment_identity_sha256"] = experiment_identity_sha256(pack["experiment"])
        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)
        assert refusal.value.stage == "identity"

    @pytest.mark.parametrize("missing", ["raw", "vector"])
    def test_omitted_raw_evidence_is_not_replaced_by_display(self, missing: str) -> None:
        """A display projection cannot certify the omitted full trajectory."""
        result = _run(LAPICQUE)
        if missing == "raw":
            result["raw"]["included"] = False
        else:
            result["raw"]["vector_snapshots_only"] = ["u"]
        with pytest.raises(ReplayRejected):
            replay_expectation(result)

    def test_export_refuses_a_pack_its_reader_cannot_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The writer enforces the same byte ceiling as the standalone reader."""
        monkeypatch.setattr("sc_neurocore.studio.replay_pack.MAX_PACK_BYTES", 16)
        with pytest.raises(ReplayRejected, match="file size limit"):
            build_replay_pack(LAPICQUE)

    def test_invalid_tolerance_is_refused_before_execution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid numerical policy must not spend compute before rejection."""
        pack = build_replay_pack(LAPICQUE)

        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("experiment executed before tolerance validation")

        monkeypatch.setattr("sc_neurocore.studio.replay_pack.run_experiment", fail)
        with pytest.raises(ReplayRejected, match="finite and non-negative"):
            replay_pack(pack, tolerance=float("nan"))

    def test_missing_full_samples_in_v2_are_refused(self) -> None:
        """v2 cannot silently downgrade its trace contract to summary-only evidence."""
        pack = build_replay_pack(LAPICQUE)
        del pack["expectation"]["states"]["v"]["samples"]
        with pytest.raises(ReplayRejected, match="missing samples"):
            verify_replay_pack(pack)

    def test_samples_must_match_their_declared_digest(self) -> None:
        """Admission refuses edited samples even if the summary fields look plausible."""
        pack = build_replay_pack(LAPICQUE)
        pack["expectation"]["states"]["v"]["samples"][10] += 0.1
        with pytest.raises(ReplayRejected, match="invalid sample evidence"):
            verify_replay_pack(pack)

    def test_legacy_pack_can_still_prove_exact_replay(self) -> None:
        """Retained v1 digests remain usable without inventing unavailable samples."""
        pack = build_replay_pack(LAPICQUE)
        pack["schema_version"] = "studio.replay-pack.v1"
        for block in pack["expectation"]["states"].values():
            del block["samples"]
        del pack["expectation"]["vector_states"]
        assert replay_pack(pack)["verdict"] == "match"

    def test_an_unsupported_schema_is_refused(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["schema_version"] = "studio.replay-pack.v99"

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)

        assert refusal.value.stage == "schema"
        assert "studio.replay-pack.v99" in refusal.value.reason

    def test_a_corrupted_pack_is_refused_by_its_own_digest(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["experiment"]["protocol"]["current"] = 999.0

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)

        assert refusal.value.stage == "schema"
        assert "does not describe its own specification" in refusal.value.reason

    def test_an_experiment_that_resolves_differently_here_is_refused(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        # Seal a specification for a different timestep, consistently digested:
        # this is what a package whose defaults changed would look like.
        other = resolve_experiment({**HODGKIN_HUXLEY, "dt": 0.02}).public
        pack["experiment"] = other
        pack["experiment_identity_sha256"] = experiment_identity_sha256(other)

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)

        assert refusal.value.stage in {"identity", "revision"}
        assert "numerical" in refusal.value.differences or "steps" in refusal.value.differences

    def test_a_model_that_is_not_installed_is_refused(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["request"]["name"] = "NoSuchNeuronExistsHere"

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)

        assert refusal.value.stage == "identity"
        assert "no longer resolves here" in refusal.value.reason

    def test_a_request_field_this_contract_does_not_execute_is_refused(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["request"]["backend"] = "rust"

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)

        assert refusal.value.stage == "request"
        assert refusal.value.differences == ("backend",)

    def test_runtime_drift_is_refused_unless_it_is_admitted(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["environment"]["package_version"] = "0.0.1-not-this-one"

        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack(pack)
        assert refusal.value.stage == "runtime"
        assert refusal.value.differences == ("package_version",)

        admitted = replay_pack(pack, allow_runtime_drift=True)
        assert admitted["verdict"] == "match"
        assert admitted["runtime_differences"] == ["package_version"]

    def test_a_malformed_document_is_refused(self) -> None:
        with pytest.raises(ReplayRejected) as refusal:
            verify_replay_pack({"schema_version": REPLAY_PACK_SCHEMA_VERSION})
        assert refusal.value.stage == "schema"

        with pytest.raises(ReplayRejected):
            verify_replay_pack(["not", "an", "object"])  # type: ignore[arg-type]

    def test_refusal_happens_before_the_experiment_runs(self, monkeypatch: Any) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        pack["environment"]["python"] = "1.0.0"

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("the experiment must not run before admission")

        monkeypatch.setattr("sc_neurocore.studio.replay_pack.run_experiment", fail)
        with pytest.raises(ReplayRejected):
            replay_pack(pack)


class TestComparisonReportsStructuralDifference:
    """Branches a digest comparison alone would never reach."""

    @pytest.mark.parametrize("samples", [["invalid"], [[1.0]], [float("nan")]])
    def test_malformed_sealed_samples_are_not_a_tolerance_match(self, samples: list[Any]) -> None:
        """The comparison API rejects nonnumeric, differently shaped and nonfinite evidence."""
        result = _run(LAPICQUE)
        expected = replay_expectation(result)
        expected["states"]["v"]["samples"] = samples
        assert compare_to_expectation(expected, result, tolerance=1)["verdict"] == "mismatch"

    def test_edited_samples_cannot_reuse_an_unchanged_digest(self) -> None:
        """Digest equality cannot bypass verification of the sealed sample payload."""
        result = _run(LAPICQUE)
        expected = replay_expectation(result)
        expected["states"]["v"]["samples"][10] += 0.1
        outcome = compare_to_expectation(expected, result)
        assert outcome["verdict"] == "mismatch"
        assert "sealed samples do not match" in outcome["differences"][0]

    @pytest.mark.parametrize("value", [[1.0], float("nan")])
    def test_snapshot_shape_and_finiteness_are_part_of_replay(self, value: object) -> None:
        """A scalar snapshot cannot silently accept a vector or nonfinite value."""
        result = _run(LAPICQUE)
        expected = replay_expectation(result)
        expected["initial_state"]["v"] = value
        outcome = compare_to_expectation(expected, result)
        assert outcome["verdict"] == "mismatch"
        assert "initial_state.v: invalid shape" in outcome["differences"][0]

    def test_a_state_the_replay_does_not_produce_is_named(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["states"]["not_a_variable"] = dict(mutated["states"]["v"])

        outcome = compare_to_expectation(mutated, result)

        assert outcome["verdict"] == "mismatch"
        assert "state not_a_variable absent from the replay" in outcome["differences"]

    def test_a_state_the_pack_does_not_carry_is_named(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        del mutated["states"]["m"]

        outcome = compare_to_expectation(mutated, result)

        assert outcome["verdict"] == "mismatch"
        assert "state m not in the pack" in outcome["differences"]

    def test_a_trace_of_a_different_length_is_reported_by_length(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["states"]["v"]["sha256"] = "0" * 64
        mutated["states"]["v"]["n_samples"] = 7

        outcome = compare_to_expectation(mutated, result)

        assert any("7 sealed" in difference for difference in outcome["differences"])

    def test_a_state_variable_present_on_one_side_only_is_reported(self) -> None:
        pack = build_replay_pack(HODGKIN_HUXLEY)
        result = _run(pack["request"])
        mutated = copy.deepcopy(pack["expectation"])
        mutated["initial_state"]["ghost"] = 0.0

        outcome = compare_to_expectation(mutated, result)

        assert "initial_state: variable ghost present on one side only" in outcome["differences"]


class TestFileHandling:
    def test_a_pack_round_trips_through_a_file(self, tmp_path: Path) -> None:
        pack = build_replay_pack(LAPICQUE)
        path = tmp_path / "pack.json"
        path.write_text(json.dumps(pack), encoding="utf-8")

        assert replay_pack(load_replay_pack(path))["verdict"] == "match"

    def test_a_directory_is_not_a_pack(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayRejected) as refusal:
            load_replay_pack(tmp_path)
        assert "not a regular file" in refusal.value.reason

    def test_a_missing_file_is_refused_without_a_traceback(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayRejected):
            load_replay_pack(tmp_path / "absent.json")

    def test_a_json_array_is_not_a_pack(self, tmp_path: Path) -> None:
        path = tmp_path / "pack.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ReplayRejected) as refusal:
            load_replay_pack(path)
        assert "must be a JSON object" in refusal.value.reason

    def test_invalid_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "pack.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ReplayRejected) as refusal:
            load_replay_pack(path)
        assert "not valid JSON" in refusal.value.reason

    def test_an_oversized_file_is_refused_before_it_is_parsed(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        path = tmp_path / "pack.json"
        path.write_text(json.dumps(build_replay_pack(LAPICQUE)), encoding="utf-8")
        monkeypatch.setattr("sc_neurocore.studio.replay_pack.MAX_PACK_BYTES", 16)
        with pytest.raises(ReplayRejected) as refusal:
            load_replay_pack(path)
        assert "the limit is 16" in refusal.value.reason


class TestPublicRunner:
    """Public CLI entrypoint and separate processes outside the checkout."""

    @pytest.mark.parametrize("mode", ["json", "mismatch", "refusal", "drift", "tolerance"])
    def test_public_cli_entrypoint_reports_its_actual_file_result(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str
    ) -> None:
        """Exercise CLI file parsing, exit status and diagnostics in the measured process."""
        pack = build_replay_pack(LAPICQUE)
        args: list[str] = []
        if mode == "json":
            args.append("--json")
        elif mode == "mismatch":
            pack["expectation"]["spike_count"] += 1
        elif mode in {"refusal", "drift"}:
            pack["environment"]["platform"] = "different-platform"
            if mode == "drift":
                args.append("--allow-runtime-drift")
        else:
            pack["expectation"]["final_state"]["v"] += 1e-9
            args.extend(["--tolerance", "1e-6"])
        path = tmp_path / "cli-pack.json"
        path.write_text(json.dumps(pack), encoding="utf-8")
        status = main([str(path), *args])
        captured = capsys.readouterr()
        if mode == "json":
            assert status == 0
            assert json.loads(captured.out)["verdict"] == "match"
        elif mode == "refusal":
            assert status == 2
            assert json.loads(captured.err)["stage"] == "runtime"
        elif mode == "mismatch":
            assert status == 1
            assert "spike_count" in captured.out
        else:
            assert status == 0
            assert (
                "runtime drift admitted: platform" if mode == "drift" else "match-within-tolerance"
            ) in captured.out

    def _run_runner(
        self, tmp_path: Path, pack: dict[str, Any], *args: str
    ) -> subprocess.CompletedProcess[str]:
        path = tmp_path / "pack.json"
        path.write_text(json.dumps(pack), encoding="utf-8")
        # Outside the checkout, with no PYTHONPATH: the runner must be reachable
        # through the installed package, not through this test's import path.
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "sc_neurocore.studio.replay_pack", str(path), *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=environment,
            timeout=900,
            check=False,
        )

    def test_the_module_replays_a_pack_and_exits_zero(self, tmp_path: Path) -> None:
        completed = self._run_runner(tmp_path, build_replay_pack(LAPICQUE))
        assert completed.returncode == 0, completed.stderr
        assert "verdict: match" in completed.stdout

    def test_a_mismatch_exits_one_and_names_the_difference(self, tmp_path: Path) -> None:
        pack = build_replay_pack(LAPICQUE)
        pack["expectation"]["spike_count"] += 1

        completed = self._run_runner(tmp_path, pack)

        assert completed.returncode == 1
        assert "verdict: mismatch" in completed.stdout
        assert "spike_count" in completed.stdout

    def test_a_refusal_exits_two_with_a_structured_reason(self, tmp_path: Path) -> None:
        pack = build_replay_pack(LAPICQUE)
        pack["environment"]["numpy"] = "0.0.0"

        completed = self._run_runner(tmp_path, pack)

        assert completed.returncode == 2
        detail = json.loads(completed.stderr)
        assert detail["error"] == "replay_refused"
        assert detail["stage"] == "runtime"
        assert detail["differences"] == ["numpy"]

    def test_the_runtime_drift_flag_admits_and_reports_it(self, tmp_path: Path) -> None:
        pack = build_replay_pack(LAPICQUE)
        pack["environment"]["platform"] = "NotThisPlatform"

        refused = self._run_runner(tmp_path, pack)
        assert refused.returncode == 2

        admitted = self._run_runner(tmp_path, pack, "--allow-runtime-drift")

        assert admitted.returncode == 0, admitted.stderr
        assert "verdict: match" in admitted.stdout
        assert "runtime drift admitted: platform" in admitted.stdout

    def test_a_tolerance_is_honoured_from_the_command_line(self, tmp_path: Path) -> None:
        pack = build_replay_pack(LAPICQUE)
        pack["expectation"]["final_state"]["v"] += 1e-9

        exact = self._run_runner(tmp_path, pack)
        assert exact.returncode == 1

        tolerated = self._run_runner(tmp_path, pack, "--tolerance", "1e-6")

        assert tolerated.returncode == 0, tolerated.stderr
        assert "match-within-tolerance" in tolerated.stdout

    def test_json_output_carries_the_full_outcome(self, tmp_path: Path) -> None:
        completed = self._run_runner(tmp_path, build_replay_pack(LAPICQUE), "--json")
        assert completed.returncode == 0, completed.stderr
        outcome = json.loads(completed.stdout)
        assert outcome["verdict"] == "match"
        assert outcome["observed"]["spike_count"] >= 0

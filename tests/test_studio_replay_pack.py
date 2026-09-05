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
        mutated = copy.deepcopy(pack["expectation"])
        mutated["states"]["v"]["sha256"] = "0" * 64
        mutated["states"]["v"]["last"] = float(mutated["states"]["v"]["last"]) + 1e-9

        outcome = compare_to_expectation(mutated, result, tolerance=1e-6)

        assert outcome["verdict"] == "match-within-tolerance"
        assert outcome["worst_state_deviation"] <= 1e-6
        assert compare_to_expectation(mutated, result, tolerance=0.0)["verdict"] == "mismatch"


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
    """The runner other people use: a subprocess, outside the checkout."""

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

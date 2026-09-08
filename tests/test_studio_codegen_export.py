# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio code export executes the experiment it came from

"""Exported code is judged by running it, never by reading it.

Every case here runs the generated script in a separate interpreter, outside
the checkout and without ``PYTHONPATH``, and compares its output with the run
the export came from. The four cases are the ones the previous export got
wrong: a model whose step takes ``drive``, a nondefault time step, a
non-constant drive protocol, and a seeded stochastic trial.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.codegen import (
    generate_experiment_script,
    generate_oneliner,
    generate_replay_script,
)
from sc_neurocore.studio.experiment_spec import resolve_experiment, run_experiment
from sc_neurocore.studio.replay_pack import build_replay_pack, pinned_request

STEP_SIGNATURE = {
    "name": "SCLapicqueLIFNeuron",
    "dt": 0.25,
    "duration": 60.0,
    "current": 1.5,
    "protocol": "ramp",
}
NONDEFAULT_DT = {
    "name": "HodgkinHuxleyNeuron",
    "dt": 0.05,
    "duration": 50.0,
    "current": 10.0,
    "protocol": "constant",
}
NONCONSTANT_PROTOCOL = {**NONDEFAULT_DT, "dt": 0.01, "protocol": "step"}
MULTI_STATE = {
    "name": "HindmarshRoseNeuron",
    "dt": 0.05,
    "duration": 200.0,
    "current": 2.0,
    "protocol": "constant",
}
SEEDED_STOCHASTIC = {
    "equations": ["dv/dt = (-(v) + I)/tau + 0.4*xi"],
    "threshold": "v > 1",
    "reset": "v = 0",
    "params": {"tau": 10.0},
    "init": {"v": 0.0},
    "dt": 0.1,
    "duration": 100.0,
    "current": 1.2,
    "seed": 4242,
}

CASES = {
    "step-signature": STEP_SIGNATURE,
    "nondefault-dt": NONDEFAULT_DT,
    "nonconstant-protocol": NONCONSTANT_PROTOCOL,
    "multi-state": MULTI_STATE,
    "seeded-stochastic": SEEDED_STOCHASTIC,
}


def _clean_environment() -> dict[str, str]:
    """Remove explicit PYTHONPATH; installed editable-package hooks remain active."""
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _execute(
    script: str, tmp_path: Path, name: str = "export.py"
) -> subprocess.CompletedProcess[str]:
    path = tmp_path / name
    path.write_text(script, encoding="utf-8")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_clean_environment(),
        timeout=900,
        check=False,
    )


def _export(request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    spec = resolve_experiment(request)
    return generate_experiment_script(spec, pinned_request(request, spec)), run_experiment(spec)


class TestExportedScriptsRun:
    @pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
    def test_the_exported_script_reproduces_the_run_it_came_from(
        self, case: str, tmp_path: Path
    ) -> None:
        script, reference = _export(CASES[case])

        completed = _execute(script, tmp_path)

        assert completed.returncode == 0, completed.stderr
        expected = (
            f"{reference['spike_count']} spikes in {reference['n_steps']} steps "
            f"at dt {reference['dt']} ms"
        )
        assert expected in completed.stdout
        assert str(reference["final_state"]) in completed.stdout

    def test_the_export_honours_the_requested_timestep(self, tmp_path: Path) -> None:
        # The model's own default is 0.01 ms; the request asks for 0.05 ms.
        # An export that dropped dt ran five times less model time and still
        # printed a plausible spike count.
        script, reference = _export(NONDEFAULT_DT)
        assert reference["dt"] == 0.05
        assert reference["n_steps"] == 1000

        completed = _execute(script, tmp_path)

        assert "at dt 0.05 ms" in completed.stdout
        assert f"in {reference['n_steps']} steps" in completed.stdout

    def test_the_export_carries_the_drive_protocol(self, tmp_path: Path) -> None:
        stepped_script, stepped = _export(NONCONSTANT_PROTOCOL)
        constant = run_experiment(
            resolve_experiment({**NONCONSTANT_PROTOCOL, "protocol": "constant"})
        )
        # The two experiments genuinely differ, so an export that silently
        # ran the constant one would be caught here.
        assert stepped["spike_count"] != constant["spike_count"]

        completed = _execute(stepped_script, tmp_path)

        assert f"{stepped['spike_count']} spikes" in completed.stdout
        assert f"{constant['spike_count']} spikes" not in completed.stdout

    def test_the_export_records_every_state_of_a_multi_state_model(self, tmp_path: Path) -> None:
        script, reference = _export(MULTI_STATE)
        assert set(reference["final_state"]) >= {"x", "y", "z"}

        completed = _execute(script, tmp_path)

        assert completed.returncode == 0, completed.stderr
        for name, value in reference["final_state"].items():
            assert f"{name!r}: {value!r}" in completed.stdout

    def test_a_seeded_stochastic_export_replays_the_same_trial(self, tmp_path: Path) -> None:
        script, reference = _export(SEEDED_STOCHASTIC)
        assert reference["spike_count"] > 0

        first = _execute(script, tmp_path, "first.py")
        second = _execute(script, tmp_path, "second.py")

        assert first.stdout == second.stdout
        assert f"{reference['spike_count']} spikes" in first.stdout

    def test_a_fresh_trial_is_exported_as_the_trial_it_drew(self, tmp_path: Path) -> None:
        request = {key: value for key, value in SEEDED_STOCHASTIC.items() if key != "seed"}
        spec = resolve_experiment({**request, "trial": "fresh"})
        sealed = pinned_request({**request, "trial": "fresh"}, spec)
        reference = run_experiment(resolve_experiment(sealed))
        script = generate_experiment_script(resolve_experiment(sealed), sealed)

        completed = _execute(script, tmp_path)

        assert "seed" in script
        assert f"{reference['spike_count']} spikes" in completed.stdout


class TestExportRefusesToLieAboutDrift:
    def test_a_changed_experiment_stops_the_script_instead_of_reporting_a_result(
        self, tmp_path: Path
    ) -> None:
        script, _ = _export(NONDEFAULT_DT)
        # A user edits the request but not the digest: the script must refuse
        # rather than print numbers under the exported experiment's name.
        edited = script.replace('"dt": 0.05', '"dt": 0.02')
        assert edited != script

        completed = _execute(edited, tmp_path)

        assert completed.returncode != 0
        assert "resolves a different experiment" in completed.stderr
        assert "spikes in" not in completed.stdout

    def test_an_unresolvable_request_fails_loudly(self, tmp_path: Path) -> None:
        script, _ = _export(NONDEFAULT_DT)
        edited = script.replace('"HodgkinHuxleyNeuron"', '"NoSuchNeuronExistsHere"')

        completed = _execute(edited, tmp_path)

        assert completed.returncode != 0
        assert "spikes in" not in completed.stdout


class TestOnelinerAndReplayScript:
    @pytest.mark.parametrize("filename", ["--pack.json", 'pack"""quoted.json'])
    def test_replay_filename_is_data_not_script_or_cli_syntax(
        self, filename: str, tmp_path: Path
    ) -> None:
        """Option-like and quote-bearing filenames remain literal replay input paths."""
        pack = build_replay_pack(STEP_SIGNATURE)
        (tmp_path / filename).write_text(json.dumps(pack), encoding="utf-8")
        completed = _execute(generate_replay_script(filename), tmp_path, "replay.py")
        assert completed.returncode == 0, completed.stderr
        assert "verdict: match" in completed.stdout

    @pytest.mark.parametrize("mode", ["missing", "json", "schema", "revision", "backend"])
    def test_replay_script_preserves_cli_refusal_exit_code(self, mode: str, tmp_path: Path) -> None:
        """Generated replay scripts preserve CLI exit2 for admission and file refusals."""
        pack = build_replay_pack(STEP_SIGNATURE)
        path = tmp_path / "replay_pack.json"
        if mode == "json":
            path.write_text("not json", encoding="utf-8")
        elif mode != "missing":
            if mode == "schema":
                pack["schema_version"] = "unsupported"
            elif mode == "revision":
                pack["request"]["dt"] = 0.1
            elif mode == "backend":
                pack["request"]["backend"] = "rust"
            path.write_text(json.dumps(pack), encoding="utf-8")
        completed = _execute(generate_replay_script(), tmp_path, "replay.py")
        assert completed.returncode == 2, completed.stderr
        assert completed.stdout == ""
        assert json.loads(completed.stderr)["error"] == "replay_refused"
        assert "Traceback" not in completed.stderr

    def test_the_oneliner_runs_the_same_experiment(self, tmp_path: Path) -> None:
        spec = resolve_experiment(STEP_SIGNATURE)
        reference = run_experiment(spec)
        oneliner = generate_oneliner(spec, pinned_request(STEP_SIGNATURE, spec))

        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", oneliner],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=_clean_environment(),
            timeout=900,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert (
            f"{reference['spike_count']} spikes in {reference['n_steps']} steps" in completed.stdout
        )

    def test_the_replay_script_verifies_a_saved_pack(self, tmp_path: Path) -> None:
        pack = build_replay_pack(STEP_SIGNATURE)
        (tmp_path / "replay_pack.json").write_text(json.dumps(pack), encoding="utf-8")

        completed = _execute(generate_replay_script(), tmp_path, "replay.py")

        assert completed.returncode == 0, completed.stderr
        assert "verdict: match" in completed.stdout

    def test_the_replay_script_reports_a_mismatch_without_pretending(self, tmp_path: Path) -> None:
        pack = build_replay_pack(STEP_SIGNATURE)
        pack["expectation"]["spike_count"] += 7
        (tmp_path / "replay_pack.json").write_text(json.dumps(pack), encoding="utf-8")

        completed = _execute(generate_replay_script(), tmp_path, "replay.py")

        assert completed.returncode == 1
        assert "verdict: mismatch" in completed.stdout
        assert "spike_count" in completed.stdout

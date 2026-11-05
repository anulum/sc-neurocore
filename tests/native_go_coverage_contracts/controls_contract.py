# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Genuine native refusal-control falsification

"""Check real profile outcomes and detect a control that accidentally restores valid metadata."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from tools.go_coverage_profiles import (
    DIAGNOSTIC_SOURCES,
    TARGET_STATEMENTS,
    compare_profiles,
    diagnostic_statements,
    read_profile,
)
from tools.go_coverage_controls import (
    native_refusal_controls,
    profile_verdict,
    validate_closed_environment_refusal,
    validate_collision_refusal,
    validate_image_refusal_result,
    validate_missing_tool_result,
)
from tools.go_coverage_records import snapshot_directories
from tools.go_mat_coverage_build import NativeGo


@pytest.mark.parametrize(("complete", "expected"), [(False, True), (True, False)])
def test_profile_control_refuses_an_expectation_opposite_to_actual_go_counters(
    native_job: Path,
    native_builder: NativeGo,
    complete: bool,
    expected: bool,
) -> None:
    """A real native verdict cannot be relabelled by changing the caller's expectation."""
    before = native_job / "artifacts/before.txt"
    after = native_job / ("artifacts/complete.txt" if complete else "artifacts/before.txt")
    with pytest.raises(RuntimeError, match="did not match"):
        profile_verdict(native_builder, before, after, "contrary-expectation", expected)
    actual = json.loads((native_builder.job / "artifacts/contrary-expectation.json").read_text())
    assert actual["passed"] is complete
    receipt = json.loads(
        (native_builder.job / "commands/contrary-expectation/receipt.json").read_text()
    )
    assert receipt["returncode"] == (0 if complete else 1)


def test_control_detects_real_go_acceptance_when_input_damage_is_reversed(
    native_job: Path,
    native_builder: NativeGo,
) -> None:
    """Pre-damaged metadata makes the control restore genuine bytes; actual Go then returns zero."""
    original = snapshot_directories(native_job, "before")[0]
    raw = native_builder.job / "pre-damaged-real-input"
    shutil.copytree(original, raw)
    metadata = next(raw.glob("covmeta.*"))
    original_bytes = metadata.read_bytes()
    damaged = bytearray(original_bytes)
    damaged[0] ^= 255
    metadata.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="accepted corrupt real metadata"):
        native_refusal_controls(
            native_builder,
            native_job / "artifacts/before.txt",
            native_job / "artifacts/complete.txt",
            raw,
        )
    restored = next((native_builder.job / "artifacts/corrupt-metadata").glob("covmeta.*"))
    assert restored.read_bytes() == original_bytes
    receipt = json.loads(
        (native_builder.job / "commands/corrupt-native-metadata/receipt.json").read_text()
    )
    assert receipt["returncode"] == 0 and receipt["process_reaped"] is True
    assert next(original.glob("covmeta.*")).read_bytes() == original_bytes


def test_original_real_refusal_receipts_remain_distinct_from_success(native_job: Path) -> None:
    """Retain actual failures and the collision case that launches no child."""
    for name, status in (
        ("missing-image", 96),
        ("wrong-digest", 1),
        ("normal-image", 97),
        ("uncovered-tag", 97),
        ("omitted-source", 1),
        ("zero-counter", 1),
    ):
        receipt = json.loads((native_job / "commands" / name / "receipt.json").read_text())
        assert receipt["returncode"] == status and receipt["process_reaped"] is True
    collision = json.loads((native_job / "artifacts/collision-control.json").read_text())
    assert collision["native_process_launched"] is False
    assert not (native_job / "commands/collision-control").exists()
    missing = json.loads((native_job / "commands/missing-native-tool/receipt.json").read_text())
    assert missing["pid"] is None and missing["returncode"] is None


@pytest.mark.parametrize("fault", ["malformed-profile", "output-parent-is-file"])
def test_public_profile_cli_retains_real_parse_and_output_creation_refusals(
    native_job: Path, native_builder: NativeGo, fault: str
) -> None:
    """The actual CLI reads damaged native data or meets a real filesystem error and returns one."""
    before = native_job / "artifacts/before.txt"
    after = native_job / "artifacts/complete.txt"
    output = native_builder.job / "artifacts/refused-profile.json"
    if fault == "malformed-profile":
        copied = native_builder.job / "artifacts/malformed-actual-profile.txt"
        lines = after.read_text().splitlines()
        assert lines[0] == "mode: atomic"
        lines[0] = "mode: invalid"
        copied.write_text("\n".join(lines) + "\n")
        after = copied
    else:
        blocked = native_builder.job / "artifacts/not-a-directory"
        blocked.write_text("The existing file must not be replaced by a result directory.\n")
        output = blocked / "result.json"
    status = native_builder.command(
        "profile-cli-refusal",
        native_builder.python_script(
            native_builder.repository / "tools/go_coverage_profiles.py",
            ["--before", str(before), "--after", str(after), "--output", str(output)],
        ),
        native_builder.job,
    )
    assert status == 1
    if fault == "malformed-profile":
        actual = json.loads(output.read_text())
        assert actual["passed"] is False
        assert actual["observed_after"] is None and actual["observation_error"]
    else:
        assert not output.exists()
        error = (native_builder.job / "commands/profile-cli-refusal/stderr.log").read_text()
        assert "Native profile result could not be created:" in error
        assert blocked.is_file()


def _rewrite_rows(source: Path, target: Path, rewrite: Callable[[str], list[str]]) -> Path:
    """Copy an actual native profile, changing only the data rows chosen by ``rewrite``."""
    header, *rows = source.read_text(encoding="utf-8").splitlines()
    assert header == "mode: atomic" and rows
    target.write_text(
        "\n".join([header, *(new for row in rows for new in rewrite(row))]) + "\n",
        encoding="utf-8",
    )
    return target


def _row_fields(row: str) -> tuple[str, str, int, int]:
    """Split one actual row into filename, geometry, statements and count."""
    location, statements, count = row.rsplit(" ", 2)
    filename, geometry = location.rsplit(":", 1)
    return filename, geometry, int(statements), int(count)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("malformed-row", "Malformed native profile row"),
        ("invalid-geometry", "Invalid native source geometry"),
        ("duplicate-block", "Duplicate native block"),
        ("changed-denominator", "statement denominator changed"),
        ("shrunk-subject", "statement scope shrunk"),
        ("executed-before-work", "already executed"),
        ("uncovered-diagnostic", "uncovered native diagnostic statements"),
    ],
)
def test_actual_profile_damage_meets_each_native_admission_refusal(
    native_job: Path, tmp_path: Path, fault: str, message: str
) -> None:
    """Each admission refusal is reached from damaged copies of the real Go profiles."""
    before = native_job / "artifacts/before.txt"
    complete = native_job / "artifacts/complete.txt"
    damaged = tmp_path / (fault + ".txt")
    subject = next(iter(TARGET_STATEMENTS))
    if fault in {"malformed-row", "invalid-geometry", "duplicate-block"}:
        first = complete.read_text(encoding="utf-8").splitlines()[1]
        filename, geometry, statements, count = _row_fields(first)
        replacement = {
            "malformed-row": [first, first.rsplit(" ", 1)[0]],
            "invalid-geometry": [
                f"{filename}:0{geometry[geometry.index('.') :]} {statements} {count}"
            ],
            "duplicate-block": [first, first],
        }[fault]
        _rewrite_rows(complete, damaged, lambda row: replacement if row == first else [row])
        with pytest.raises(ValueError, match=message):
            read_profile(damaged)
    elif fault == "changed-denominator":
        first = complete.read_text(encoding="utf-8").splitlines()[1]
        filename, geometry, statements, count = _row_fields(first)
        changed = f"{filename}:{geometry} {statements + 1} {count}"
        _rewrite_rows(complete, damaged, lambda row: [changed] if row == first else [row])
        with pytest.raises(ValueError, match=message):
            compare_profiles(before, damaged)
    elif fault == "shrunk-subject":

        def emptied(row: str) -> list[str]:
            filename, geometry, _, count = _row_fields(row)
            return [f"{filename}:{geometry} 0 {count}" if filename == subject else row]

        shrunk_before = _rewrite_rows(before, tmp_path / "shrunk-before.txt", emptied)
        _rewrite_rows(complete, damaged, emptied)
        with pytest.raises(ValueError, match=message + ": " + subject):
            compare_profiles(shrunk_before, damaged)
    elif fault == "executed-before-work":
        with pytest.raises(ValueError, match=message + ": " + subject):
            compare_profiles(complete, complete)
    else:
        diagnostic = DIAGNOSTIC_SOURCES[0]

        def unexecuted(row: str) -> list[str]:
            filename, geometry, statements, _ = _row_fields(row)
            return [f"{filename}:{geometry} {statements} 0" if filename == diagnostic else row]

        assert diagnostic_statements(complete)
        _rewrite_rows(complete, damaged, unexecuted)
        with pytest.raises(ValueError, match=message + ": " + diagnostic):
            diagnostic_statements(damaged)
    assert read_profile(complete) and read_profile(before)


@pytest.mark.parametrize("fault", ["accepted", "other-error", "existing-output"])
def test_collision_evidence_refuses_success_wrong_cause_or_an_existing_receipt(
    native_builder: NativeGo, fault: str
) -> None:
    """Actual command outcomes cannot be relabelled as a successful no-overwrite control."""
    collision = native_builder.job / "commands/unlaunched/receipt.json"
    caught: ValueError | None = None
    if fault == "accepted":
        assert (
            native_builder.command(
                "actual-success", [str(native_builder.go), "version"], native_builder.source
            )
            == 0
        )
    else:
        try:
            native_builder.command(
                "../invalid" if fault == "other-error" else "collision",
                [str(native_builder.go), "version"],
                native_builder.source,
                outputs=() if fault == "other-error" else (collision,),
            )
        except ValueError as error:
            caught = error
        assert caught is not None
        if fault == "existing-output":
            collision.parent.mkdir()
            collision.write_text("Existing bytes must survive the refused control.\n")
    before = collision.read_bytes() if collision.exists() else None
    with pytest.raises(RuntimeError, match="overwrite|not refused"):
        validate_collision_refusal(caught, collision)
    assert (collision.read_bytes() if collision.exists() else None) == before


@pytest.mark.parametrize("field", ["pid", "returncode"])
def test_missing_tool_evidence_refuses_fields_borrowed_from_a_real_successful_command(
    native_job: Path, native_builder: NativeGo, field: str
) -> None:
    """Only a damaged copy changes; the original genuine spawn-failure receipt remains null."""
    with pytest.raises(FileNotFoundError) as error:
        native_builder.command(
            "actual-missing", [str(native_builder.job / "absent")], native_builder.job
        )
    path = native_builder.job / "commands/actual-missing/receipt.json"
    original = path.read_bytes()
    receipt = cast(dict[str, object], json.loads(original))
    success = json.loads((native_job / "commands/go-format/receipt.json").read_text())
    assert success[field] is not None
    receipt[field] = success[field]
    with pytest.raises(RuntimeError, match="process status was invented"):
        validate_missing_tool_result(error.value, receipt)
    assert path.read_bytes() == original


def test_missing_tool_control_detects_an_actual_available_pinned_go_executable(
    native_job: Path, native_builder: NativeGo
) -> None:
    """The real SDK, exposed at the control path, must invalidate the missing-tool expectation."""
    (native_builder.job / "absent-go").symlink_to(native_builder.go)
    with pytest.raises(RuntimeError, match="unavailable native executable was accepted"):
        native_refusal_controls(
            native_builder,
            native_job / "artifacts/before.txt",
            native_job / "artifacts/complete.txt",
            snapshot_directories(native_job, "before")[0],
        )
    receipt = json.loads(
        (native_builder.job / "commands/missing-native-tool/receipt.json").read_text()
    )
    assert receipt["pid"] > 0 and receipt["process_reaped"] is True
    assert receipt["returncode"] is not None and receipt["operation_error"] is None


@pytest.mark.parametrize(
    "fault", ["status", "missing-stream", "duplicate-stream", "event", "error", "stderr"]
)
def test_damaged_actual_native_refusal_evidence_cannot_satisfy_the_control(
    native_job: Path, tmp_path: Path, fault: str
) -> None:
    """Refuse damaged original streams and real outcomes from another phase."""
    phase, expected, reason = "normal-image", 97, "write_coverage_snapshot"
    actual = json.loads((native_job / "commands/normal-image/receipt.json").read_text())[
        "returncode"
    ]
    if fault == "status":
        other = json.loads((native_job / "commands/wrong-digest/receipt.json").read_text())
        actual = other["returncode"]
    elif fault == "stderr":
        phase, expected, reason = (
            "wrong-digest",
            1,
            "Diagnostic image bytes or metadata changed before load",
        )
        actual = json.loads((native_job / "commands/wrong-digest/receipt.json").read_text())[
            "returncode"
        ]
        output = tmp_path / "commands/wrong-digest/stderr.log"
        output.parent.mkdir(parents=True)
        shutil.copyfile(native_job / "commands/go-format/stdout.log", output)
    else:
        original = next((native_job / "process-counters/normal-image").glob("*/observations.jsonl"))
        if fault != "missing-stream":
            output = (
                tmp_path / "process-counters/normal-image" / original.parent.name / original.name
            )
            output.parent.mkdir(parents=True)
            shutil.copyfile(original, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if fault == "duplicate-stream":
                duplicate = (
                    output.parent.parent / (original.parent.name + "-copied") / original.name
                )
                duplicate.parent.mkdir()
                shutil.copyfile(original, duplicate)
            elif fault == "event":
                before = next((native_job / "process-counters/before").glob("*/observations.jsonl"))
                finished = json.loads(before.read_text().splitlines()[-1])
                assert finished["event"] == "finished"
                rows[-1]["event"] = finished["event"]
                output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            else:
                rows[-1]["error"] = (native_job / "commands/wrong-digest/stderr.log").read_text()
                output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(
        RuntimeError,
        match="status differs|observer record|absent diagnostic export|unrelated reason",
    ):
        validate_image_refusal_result(tmp_path, phase, expected, actual, reason)


@pytest.mark.parametrize("fault", ["status", "reason"])
def test_actual_other_failures_cannot_claim_closed_environment_refusal(
    native_job: Path, fault: str
) -> None:
    """Retain actual refusal receipts while presenting a genuine unrelated status or error text."""
    label = "closed-environment-godebug"
    status = json.loads((native_job / "commands" / label / "receipt.json").read_text())[
        "returncode"
    ]
    error = (native_job / "commands" / label / "stderr.log").read_text()
    if fault == "status":
        status = json.loads((native_job / "commands/wrong-digest/receipt.json").read_text())[
            "returncode"
        ]
    else:
        error = (native_job / "commands/missing-image/stderr.log").read_text()
    with pytest.raises(RuntimeError, match="unreviewed consumer environment was not refused"):
        validate_closed_environment_refusal(status, error)

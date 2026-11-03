# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native Go coverage refusal controls

"""Exercise native tools and images without replacing their real refusal outcomes."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import shutil

from tools.go_coverage_profiles import TARGET_STATEMENTS
from tools.go_mat_coverage_build import NativeGo, digest, save


def validate_collision_refusal(error: ValueError | None, collision: Path) -> None:
    """Require an actual namespace refusal before accepting that no receipt was overwritten."""
    if error is None:
        raise RuntimeError("A native command output was allowed to overwrite its receipt")
    if "collides" not in str(error) or collision.exists():
        raise RuntimeError(
            "Native output collision was not refused before any overwrite"
        ) from error


def validate_missing_tool_result(
    error: FileNotFoundError | None, receipt: dict[str, object]
) -> None:
    """Conserve the real spawn exception and the absence of any invented PID or exit status."""
    if error is None:
        raise RuntimeError("An unavailable native executable was accepted")
    if receipt["pid"] is not None or receipt["returncode"] is not None:
        raise RuntimeError("A process status was invented for an unavailable executable") from error


def validate_image_refusal_result(
    job: Path, phase: str, status: int, actual: int, reason: str
) -> None:
    """Read the real refusal stream and stderr, requiring the exact phase status and cause."""
    if actual != status:
        raise RuntimeError(f"Actual {phase} refusal status differs: {actual}")
    if phase in {"normal-image", "uncovered-tag"}:
        streams = list((job / "process-counters" / phase).glob("*/observations.jsonl"))
        if len(streams) != 1:
            raise RuntimeError("Ordinary-image refusal omitted its actual observer record")
        rows = [json.loads(line) for line in streams[0].read_text().splitlines()]
        if rows[-1].get("event") != "snapshot_failure" or reason not in rows[-1].get("error", ""):
            raise RuntimeError(
                "Ordinary-image refusal did not identify its absent diagnostic export"
            )
    elif reason not in (job / "commands" / phase / "stderr.log").read_text():
        raise RuntimeError("The native image control failed for an unrelated reason")


def validate_closed_environment_refusal(status: int, error: str) -> None:
    """Require a real closed-environment refusal rather than an unrelated failed command."""
    if (
        status != 96
        or "Actual consumer keys or values differ from the closed environment" not in error
    ):
        raise RuntimeError("An actual unreviewed consumer environment was not refused")


def profile_verdict(builder: NativeGo, before: Path, after: Path, label: str, passed: bool) -> Path:
    """Require the actual public profile CLI's positive or refusal OS result."""
    output = builder.job / "artifacts" / (label + ".json")
    status = builder.command(
        label,
        builder.python_script(
            builder.repository / "tools/go_coverage_profiles.py",
            [
                "--before",
                str(before),
                "--after",
                str(after),
                "--output",
                str(output),
            ],
        ),
        builder.job,
        outputs=(output,),
    )
    result = json.loads(output.read_text())
    if status != (0 if passed else 1) or result.get("passed") is not passed:
        raise RuntimeError("Actual public profile verdict did not match its native inputs")
    return output


def native_refusal_controls(builder: NativeGo, before: Path, complete: Path, raw: Path) -> None:
    """Exercise real metadata, text, missing-tool and output-collision refusals."""
    corrupt = builder.job / "artifacts/corrupt-metadata"
    shutil.copytree(raw, corrupt)
    metadata = sorted(corrupt.glob("covmeta.*"))[0]
    data = bytearray(metadata.read_bytes())
    data[0] ^= 255
    metadata.write_bytes(data)
    status = builder.command(
        "corrupt-native-metadata",
        [
            str(builder.go),
            "tool",
            "covdata",
            "debugdump",
            "-i=" + str(corrupt),
        ],
        builder.source,
    )
    if status == 0:
        raise RuntimeError("The native Go reader accepted corrupt real metadata")
    for damage in ("omitted-source", "zero-counter"):
        lines = complete.read_text().splitlines()
        if damage == "omitted-source":
            lines = [line for line in lines if "/services/mat.go:" not in line]
        else:
            index = next(
                index
                for index, line in enumerate(lines[1:], 1)
                if line.rsplit(":", 1)[0] in TARGET_STATEMENTS and int(line.split()[-2]) > 0
            )
            geometry, statements, _ = lines[index].split()
            lines[index] = geometry + " " + statements + " 0"
        path = builder.job / "artifacts" / (damage + ".txt")
        path.write_text("\n".join(lines) + "\n")
        profile_verdict(builder, before, path, damage, False)
    collision = builder.job / "commands/collision-control/receipt.json"
    argv = [
        str(builder.go),
        "tool",
        "covdata",
        "textfmt",
        "-i=" + str(builder.job / "artifacts/before-merged"),
        "-o=" + str(collision),
    ]
    collision_error: ValueError | None = None
    try:
        builder.command("collision-control", argv, builder.source, outputs=(collision,))
    except ValueError as error:
        collision_error = error
    validate_collision_refusal(collision_error, collision)
    save(
        builder.job / "artifacts/collision-control.json",
        {
            "argv": argv,
            "refusal": str(collision_error),
            "native_process_launched": False,
            "existing_artifacts_preserved": True,
        },
    )
    missing_error: FileNotFoundError | None = None
    try:
        builder.command("missing-native-tool", [str(builder.job / "absent-go")], builder.source)
    except FileNotFoundError as error:
        missing_error = error
    receipt = json.loads((builder.job / "commands/missing-native-tool/receipt.json").read_text())
    validate_missing_tool_result(missing_error, receipt)


def image_refusal_controls(
    builder: NativeGo,
    images: list[dict[str, str]],
    built: dict[str, Path],
    preload: Path,
    environment_for: Callable[[str, str], dict[str, str]],
) -> None:
    """Require genuine missing, changed-digest and ordinary-image load refusals."""
    for phase, manifest, status, reason in (
        ("missing-image", "images-missing-image.json", 96, "FileNotFoundError"),
        (
            "wrong-digest",
            "images-wrong-digest.json",
            1,
            "Diagnostic image bytes or metadata changed before load",
        ),
        ("normal-image", "images-normal.json", 97, "write_coverage_snapshot"),
        ("uncovered-tag", "images-uncovered-tag.json", 97, "Native Go counter snapshot refused"),
    ):
        declarations = [dict(row) for row in images]
        for row in declarations:
            if row["backend"] != "go":
                continue
            if phase == "missing-image":
                row["path"] = str(
                    builder.job / "absent" / row["model"] / ("lib" + row["model"] + ".so")
                )
            elif phase == "wrong-digest":
                row["sha256"] = "0" * 64
            else:
                variant = "tagged-uninstrumented" if phase == "uncovered-tag" else "normal"
                path = built[variant + "/" + row["model"]]
                row.update(path=str(path), sha256=digest(path))
        save(builder.job / manifest, declarations)
        environment = environment_for(phase, manifest)
        actual = builder.command(
            phase,
            builder.python_script(
                preload,
                [
                    str(builder.repository),
                    str(builder.job),
                    manifest,
                ],
            ),
            builder.job,
            environment=environment,
        )
        validate_image_refusal_result(builder.job, phase, status, actual, reason)
    for name, value in (
        ("LD_LIBRARY_PATH", "/absent"),
        ("PYTHONHOME", "/absent"),
        ("GODEBUG", "panicnil=1"),
    ):
        # -I prevents Python itself from acting on the poison before the real
        # bootstrap verifies the actual closed mapping. Nothing is inherited.
        label = "closed-environment-" + name.lower().replace("_", "-")
        environment = environment_for("before", "images.json")
        environment = {**environment, name: value}
        actual = builder.command(
            label,
            builder.python_script(
                preload,
                [
                    str(builder.repository),
                    str(builder.job),
                    "images.json",
                ],
            ),
            builder.job,
            environment=environment,
        )
        error = (builder.job / "commands" / label / "stderr.log").read_text()
        validate_closed_environment_refusal(actual, error)

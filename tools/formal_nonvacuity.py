# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native assertion-enable non-vacuity

"""Check assertion-enable reachability using the primary proof's prepared model."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tools.formal_assertion_enables import enable_model, prepared_top
from tools.formal_evidence_validation import digest, object_fields, require

_OPTIONS = """
import json, sys
sys.path.insert(0, sys.argv[1])
from sby_core import SbyConfig
def error(message):
    raise ValueError(message)
config = SbyConfig()
config.error = error
config.parse_config(sys.stdin)
print(json.dumps({"options": config.options, "engines": config.engines}))
"""


def native_configuration(text: str) -> dict[str, object]:
    """Parse an SBY-rendered task configuration with its installed native parser.

    Parameters
    ----------
    text : str
        Task-specific native ``--dumpcfg`` output or retained ``config.sby``.

    Returns
    -------
    dict of str to object
        Native options and engines, without reimplementing the SBY grammar.

    Raises
    ------
    ValueError
        If the native executable or parsed configuration is unavailable.
    subprocess.SubprocessError
        If the native parser rejects the configuration.
    """
    sby = shutil.which("sby")
    require(sby is not None, "Native SBY is unavailable.")
    library = Path(str(sby)).resolve().parent.parent / "share/yosys/python3"
    return object_fields(
        json.loads(
            subprocess.run(
                [sys.executable, "-c", _OPTIONS, str(library)],
                input=text,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        )
    )


def check_task(workdir: Path, destination: Path) -> dict[str, object]:
    """Cover every assertion enable at the primary task's original depth.

    Parameters
    ----------
    workdir : Path
        Successful native SBY task directory, including its original
        ``config.sby``, ``status`` and ``model/design_prep.il``.
    destination : Path
        New auxiliary output directory. The caller's existing cgroup and
        timeout must encompass both the primary and this check.

    Returns
    -------
    dict of str to object
        Bound model hashes, original task options, assertion identities,
        native cover outcome and each first reached step.

    Raises
    ------
    ValueError
        If primary success, native configuration or non-vacuity is absent.
    OSError
        If native tools or retained primary evidence are unavailable.
    subprocess.SubprocessError
        If a native model conversion or cover invocation fails.
    """
    require((workdir / "status").read_text().split()[0] == "PASS", "Primary is not PASS.")
    return _audit_prepared(workdir, destination, assertions_only=False)


def check_prepared_assertions(workdir: Path, destination: Path) -> dict[str, object]:
    """Audit retained assertion enables without assigning a primary verdict.

    Parameters
    ----------
    workdir : Path
        Native task directory containing its unchanged ``config.sby`` and
        ``model/design_prep.il``. The caller separately validates the primary
        outcome, source and model custody; a PASS status file is not required.
    destination : Path
        New auxiliary output directory within the caller's bounded audit
        service. Only assertion-enable covers run; the primary is not repeated.

    Returns
    -------
    dict of str to object
        Bound enable witnesses and audit verdict, never a primary proof result.

    Raises
    ------
    ValueError
        If the task is not bmc/prove or its assertion-enable audit is invalid.
    OSError
        If required native tools or retained model files are unavailable.
    subprocess.SubprocessError
        If native model conversion or configuration parsing fails.
    """
    return _audit_prepared(workdir, destination, assertions_only=True)


def _audit_prepared(
    workdir: Path, destination: Path, *, assertions_only: bool
) -> dict[str, object]:
    """Cover the prepared task's actual goals under its unchanged assumptions."""
    sby = shutil.which("sby")
    require(sby is not None, "Native SBY is unavailable.")
    native_sby = Path(str(sby)).resolve()
    config = workdir / "config.sby"
    parsed = native_configuration(config.read_text())
    options = object_fields(parsed.get("options"))
    mode, depth_text = options.get("mode"), options.get("depth")
    require(mode in ("bmc", "prove", "cover"), "Unsupported native proof mode.")
    require(
        not assertions_only or mode in ("bmc", "prove"),
        "Prepared assertion audits require bmc/prove; original cover tasks require primary PASS.",
    )
    require(isinstance(depth_text, str) and depth_text.isdecimal(), "Explicit depth is required.")
    depth = int(str(depth_text))
    require(depth > 0, "Native depth must be positive.")
    prepared = workdir / "model/design_prep.il"
    original_sha = digest(prepared)
    destination.mkdir(parents=True, exist_ok=False)
    original_json = destination / "prepared.json"
    # Quote Yosys command arguments independently of shell syntax.
    script = f"read_rtlil {json.dumps(str(prepared.resolve()))}; flatten; write_json prepared.json"
    conversion = subprocess.run(
        ["yosys", "-Q", "-T", "-p", script],
        cwd=destination,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (destination / "model-conversion.log").write_text(conversion.stdout + conversion.stderr)
    require(digest(prepared) == original_sha, "Primary prepared model changed during the audit.")
    base: dict[str, object] = {
        "schema_version": 1,
        "check": "assertion-enable non-vacuity",
        "mode": mode,
        "depth": depth,
        "config_sha256": digest(config),
        "design_prep_sha256": original_sha,
        "prepared_json_sha256": digest(original_json),
        "sby_sha256": digest(native_sby),
        "implementation_sha256": digest(Path(__file__)),
    }
    if mode == "cover":
        design = object_fields(json.loads(original_json.read_text()))
        _top, cells = prepared_top(design)
        covers = [
            name for name, cell in cells.items() if object_fields(cell).get("type") == "$cover"
        ]
        require(bool(covers), "Native cover task has zero actual cover goals.")
        base.update(verdict="NONVACUOUS", original_cover_goals=covers)
        return base
    model = enable_model(original_json.read_text())
    auxiliary = destination / "enables.json"
    auxiliary.write_text(model.serialized)
    engines = object_fields(parsed.get("engines"))
    choices = engines.get("null", engines.get(str(mode)))
    if not isinstance(choices, list) or not choices:
        raise ValueError("Native engines are missing.")
    engine = choices[0]
    require(
        isinstance(engine, list)
        and bool(engine)
        and all(isinstance(token, str) for token in engine)
        and engine[0] == "smtbmc",
        "Assertion-enable covers require the task's native smtbmc engine.",
    )
    settings = ["mode cover", f"depth {depth}", "skip_prep on", "witrename off"]
    if "skip" in options:
        settings.append("skip " + str(options["skip"]))
    cover_config = destination / "enables.sby"
    cover_config.write_text(
        "[options]\n"
        + "\n".join(settings)
        + "\n[engines]\n"
        + " ".join(str(token) for token in engine)
        + "\n[script]\nread_json enables.json\nhierarchy -check -top "
        + model.top
        + "\n[files]\nenables.json\n"
    )
    result = subprocess.run(
        [str(native_sby), "-d", "cover", "enables.sby"],
        cwd=destination,
        capture_output=True,
        text=True,
        check=False,
    )
    log = result.stdout + result.stderr
    (destination / "cover.log").write_text(log)
    status_file = destination / "cover/status"
    status = status_file.read_text().split()[0] if status_file.is_file() else "ERROR"
    reached: dict[str, int] = {}
    for line in log.splitlines():
        match = re.search(r"Reached cover statement in step (\d+) at ", line)
        if match:
            for goal in model.goals:
                if re.search(r"\b" + re.escape(goal) + r"\b", line):
                    reached[goal] = min(reached.get(goal, depth), int(match[1]))
    complete = set(reached) == set(model.goals)
    qualified = (
        result.returncode == 0
        and status == "PASS"
        and "DONE (PASS, rc=0)" in log
        and complete
        and all(0 <= step < depth for step in reached.values())
    )
    base.update(
        verdict="NONVACUOUS" if qualified else "ERROR",
        native_status=status,
        native_exit_code=result.returncode,
        auxiliary_json_sha256=digest(auxiliary),
        auxiliary_config_sha256=digest(cover_config),
        auxiliary_log_sha256=digest(destination / "cover.log"),
        assertions=model.goals,
        first_reached_steps=reached,
        assertion_enable_status="REACHED" if complete else "UNREACHED",
        unreached_assertions=sorted(set(model.goals) - set(reached)),
        transform_sha256=digest(Path(__file__).with_name("formal_assertion_enables.py")),
    )
    return base

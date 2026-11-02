# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native assertion-enable covers

"""Prove real assertion enables and reject a permanently disabled native assertion."""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from tests.formal_native_evidence_support import collect_native
from tools.formal_nonvacuity import check_prepared_assertions, check_task


@pytest.fixture(scope="module")
def primary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Retain an actual successful catalogue proof for all model audit controls."""
    root = tmp_path_factory.mktemp("enable-native")
    collect_native(root)
    return root / "proof"


def test_same_native_model_reaches_every_assertion_enable(primary: Path, tmp_path: Path) -> None:
    """Bind the exact prepared model and report a reached step below its own depth."""
    result = check_task(primary, tmp_path / "positive")
    assert result["verdict"] == "NONVACUOUS"
    assert result["native_status"] == "PASS"
    steps = result["first_reached_steps"]
    assert isinstance(steps, dict) and steps
    assert all(0 <= step < int(str(result["depth"])) for step in steps.values())
    repeat = check_task(primary, tmp_path / "repeat")
    assert repeat["design_prep_sha256"] == result["design_prep_sha256"]
    assert repeat["auxiliary_json_sha256"] == result["auxiliary_json_sha256"]
    assert repeat["first_reached_steps"] == result["first_reached_steps"]


def test_prepared_audit_does_not_qualify_an_incomplete_primary(
    primary: Path, tmp_path: Path
) -> None:
    """A retained model can be audited while the ordinary PASS guard stays closed."""
    copied = tmp_path / "incomplete-primary"
    copied.mkdir()
    (copied / "model").mkdir()
    for name in ("config.sby", "model/design_prep.il"):
        shutil.copyfile(primary / name, copied / name)
    # Test-only outcome metadata; this does not claim a measured primary timeout.
    (copied / "status").write_text("TIMEOUT\n")
    with pytest.raises(ValueError, match="Primary is not PASS"):
        check_task(copied, tmp_path / "ordinary")
    result = check_prepared_assertions(copied, tmp_path / "prepared-only")
    assert result["verdict"] == "NONVACUOUS"
    assert result["native_status"] == "PASS"
    assert (copied / "status").read_text() == "TIMEOUT\n"


def test_prepared_assertion_audit_refuses_original_cover_mode(
    primary: Path, tmp_path: Path
) -> None:
    """A cover task needs its successful native primary, not just a count of goals."""
    check_task(primary, tmp_path / "positive-cover")
    with pytest.raises(ValueError, match="original cover tasks require primary PASS"):
        check_prepared_assertions(tmp_path / "positive-cover/cover", tmp_path / "refused")


def test_constant_false_enable_is_vacuous(primary: Path, tmp_path: Path) -> None:
    """The native solver must leave a deliberately disabled assertion's cover unreached."""
    copied = tmp_path / "disabled-primary"
    shutil.copytree(primary, copied)
    original = tmp_path / "prepared.json"
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_rtlil {copied}/model/design_prep.il; write_json {original}",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    design = json.loads(original.read_text())
    mutations = 0
    for module in design["modules"].values():
        for cell in module["cells"].values():
            if cell["type"] == "$assert":
                cell["connections"]["EN"] = ["0"]
                mutations += 1
    assert mutations > 0
    original.write_text(json.dumps(design))
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_json {original}; write_rtlil {copied}/model/design_prep.il",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    result = check_task(copied, tmp_path / "negative")
    assert result["verdict"] == "ERROR"
    assert result["native_status"] == "FAIL"
    assert result["first_reached_steps"] == {}


def test_one_reached_enable_cannot_qualify_a_mixed_native_model(
    primary: Path, tmp_path: Path
) -> None:
    """Audit every goal independently when native SBY also reports an unreached cover."""
    copied = tmp_path / "mixed-primary"
    shutil.copytree(primary, copied)
    serialized = tmp_path / "mixed.json"
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_rtlil {copied}/model/design_prep.il; flatten; write_json {serialized}",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    design = json.loads(serialized.read_text())
    for module in design["modules"].values():
        original = next(
            (cell for cell in module["cells"].values() if cell["type"] == "$assert"), None
        )
        if original is not None:
            disabled = deepcopy(original)
            disabled["connections"]["EN"] = ["0"]
            module["cells"]["constant_false_assertion"] = disabled
    serialized.write_text(json.dumps(design))
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_json {serialized}; write_rtlil {copied}/model/design_prep.il",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    result = check_task(copied, tmp_path / "mixed-audit")
    assert result["verdict"] == "ERROR"
    assert result["assertion_enable_status"] == "UNREACHED"
    assertions, steps, unreached = (
        result["assertions"],
        result["first_reached_steps"],
        result["unreached_assertions"],
    )
    assert isinstance(assertions, dict) and len(assertions) == 2
    assert isinstance(steps, dict) and len(steps) == 1
    assert isinstance(unreached, list) and len(unreached) == 1


def test_native_cover_task_uses_nonzero_original_goals(primary: Path, tmp_path: Path) -> None:
    """An actual reached native cover qualifies without adding another auxiliary run."""
    positive = tmp_path / "positive"
    check_task(primary, positive)
    result = check_task(positive / "cover", tmp_path / "original-cover-audit")
    assert result["verdict"] == "NONVACUOUS"
    assert result["original_cover_goals"]


@pytest.mark.parametrize("defect", ["unhandled-check", "unused-goal"])
def test_original_cover_task_refuses_unaudited_or_unused_goals(
    primary: Path, tmp_path: Path, defect: str
) -> None:
    """Only supported goals in the actual top can qualify a native cover task."""
    positive = tmp_path / "positive"
    check_task(primary, positive)
    copied = positive / "cover"
    serialized = tmp_path / "native-cover.json"
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_rtlil {copied}/model/design_prep.il; write_json {serialized}",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    design = json.loads(serialized.read_text())
    top = next(
        module
        for module in design["modules"].values()
        if module["attributes"].get("top") == "00000000000000000000000000000001"
    )
    name, goal = next(
        (name, cell) for name, cell in top["cells"].items() if cell["type"] == "$cover"
    )
    if defect == "unhandled-check":
        extra = deepcopy(goal)
        extra["type"] = "$fair"
        top["cells"]["unaudited_fairness"] = extra
    else:
        unused = deepcopy(top)
        del unused["attributes"]["top"]
        design["modules"]["unused_module_with_cover"] = unused
        del top["cells"][name]
    serialized.write_text(json.dumps(design))
    subprocess.run(
        [
            "yosys",
            "-Q",
            "-T",
            "-p",
            f"read_json {serialized}; write_rtlil {copied}/model/design_prep.il",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    with pytest.raises(ValueError, match="unsupported formal|zero actual cover"):
        check_task(copied, tmp_path / "refused-audit")


def test_skip_is_retained_at_the_native_cover_boundary(primary: Path, tmp_path: Path) -> None:
    """The auxiliary checks the original task's nonzero starting step."""
    copied = tmp_path / "primary"
    shutil.copytree(primary, copied)
    config = copied / "config.sby"
    config.write_text(config.read_text().replace("[options]", "[options]\nskip 1"))
    result = check_task(copied, tmp_path / "audit")
    assert result["verdict"] == "NONVACUOUS"
    assert "skip 1" in (tmp_path / "audit/enables.sby").read_text()


@pytest.mark.parametrize(
    "defect", ["status", "depth", "mode", "zero-cover", "missing-model", "engines"]
)
def test_missing_native_qualification_refuses(primary: Path, tmp_path: Path, defect: str) -> None:
    """Incomplete or unsupported retained evidence cannot become an enabled proof."""
    copied = tmp_path / "primary"
    shutil.copytree(primary, copied)
    if defect == "status":
        (copied / "status").write_text("FAIL\n")
    elif defect in ("depth", "mode", "zero-cover", "engines"):
        config = copied / "config.sby"
        text = config.read_text()
        if defect == "depth":
            text = text.replace("depth ", "depth invalid")
        elif defect == "engines":
            text = text.replace("smtbmc z3", "")
        else:
            text = text.replace(
                "mode bmc", "mode " + ("cover" if defect == "zero-cover" else "live")
            )
        config.write_text(text)
    else:
        (copied / "model/design_prep.il").rename(copied / "model/retained-prep.il")
    with pytest.raises((ValueError, OSError)):
        check_task(copied, tmp_path / "audit")

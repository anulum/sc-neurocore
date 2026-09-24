# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Studio verified-readiness seal

"""The seal an installation serves is the checkout's own derivation, and says it is sealed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.neurons.readiness import REPO_ROOT
from sc_neurocore.studio.model_catalogue import (
    _receipt_verified_detail,
    _verified_detail,
    readiness_seal_payload,
)
from sc_neurocore.studio.readiness_seal import (
    SEAL_PATH,
    SEAL_SCHEMA,
    SOURCE_RECEIPTS,
    SOURCE_SEALED,
    SOURCE_UNSEALED,
    build_seal,
    checkout_available,
    render_seal,
    sealed_detail,
)

TOOL = REPO_ROOT / "tools" / "studio_readiness_seal.py"


def _tool():
    spec = importlib.util.spec_from_file_location("studio_readiness_seal", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_seal_is_the_checkouts_derivation_now() -> None:
    """A seal that no longer matches the receipts fails here, before it ships."""
    assert SEAL_PATH.read_text(encoding="utf-8") == render_seal(readiness_seal_payload())


def test_the_seal_is_package_data() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"studio/verified_readiness.json",' in pyproject


def test_a_checkout_re_derives_and_says_so() -> None:
    assert checkout_available(REPO_ROOT)
    detail = _verified_detail("LapicqueNeuron")
    assert detail["source"] == SOURCE_RECEIPTS
    assert {k: v for k, v in detail.items() if k != "source"} == _receipt_verified_detail(
        "LapicqueNeuron"
    )


def test_a_tree_without_tests_or_descriptors_is_not_a_checkout(tmp_path: Path) -> None:
    assert not checkout_available(tmp_path)
    (tmp_path / "tests").mkdir()
    assert not checkout_available(tmp_path)


def test_a_sealed_model_is_served_as_sealed() -> None:
    sealed = sealed_detail("LapicqueNeuron")
    assert sealed["source"] == SOURCE_SEALED
    assert {k: v for k, v in sealed.items() if k != "source"} == _receipt_verified_detail(
        "LapicqueNeuron"
    )


def test_a_model_or_seal_that_is_absent_is_unverified_with_the_reason(tmp_path: Path) -> None:
    missing = sealed_detail("LapicqueNeuron", tmp_path / "absent.json")
    assert missing["source"] == SOURCE_UNSEALED
    assert (missing["science_tier"], missing["silicon_tier"], missing["facets"]) == (0, None, [])
    assert "carries no readiness seal" in missing["unsealed_reason"]

    seal = tmp_path / "seal.json"
    seal.write_text(render_seal(build_seal([], _receipt_verified_detail)), encoding="utf-8")
    unknown = sealed_detail("LapicqueNeuron", seal)
    assert unknown["source"] == SOURCE_UNSEALED
    assert "holds no record of this model" in unknown["unsealed_reason"]


def test_a_file_that_is_not_a_seal_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "something-else", "models": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match=SEAL_SCHEMA):
        sealed_detail("LapicqueNeuron", other)


def test_the_tool_checks_the_tracked_seal() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_tool_names_a_missing_or_stale_seal(tmp_path: Path) -> None:
    tool = _tool()
    assert tool.seal_problems(tmp_path / "absent.json") == ["missing readiness seal: absent.json"]
    stale = tmp_path / "verified_readiness.json"
    stale.write_text("{}\n", encoding="utf-8")
    assert tool.seal_problems(stale) == [
        "stale readiness seal: verified_readiness.json (run tools/studio_readiness_seal.py --write)"
    ]
    assert tool.seal_problems() == []

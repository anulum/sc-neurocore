# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine requirement gate contract

"""The engine import gate skips locally and never masks hosted CI."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from tests.engine_requirement import require_engine

_ROOT = Path(__file__).resolve().parents[1]


def test_require_engine_returns_the_module_when_present() -> None:
    module = require_engine()
    assert module.__name__ == "sc_neurocore_engine"


def test_require_engine_skips_locally_when_module_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_ENGINE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_engine("sc_neurocore_engine_that_does_not_exist")


def test_require_engine_hard_fails_when_ci_requires_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_ENGINE", "1")
    with pytest.raises(ModuleNotFoundError):
        require_engine("sc_neurocore_engine_that_does_not_exist")


def test_hosted_ci_exports_the_engine_requirement() -> None:
    """The test job must forbid the skip path so CI cannot false-green."""
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "SC_NEUROCORE_REQUIRE_ENGINE" in workflow


def test_no_binding_test_imports_the_engine_unguarded() -> None:
    """Every module-level engine import sits behind a skip-capable gate."""
    engine_import = re.compile(
        r"^(import sc_neurocore_engine\b|from sc_neurocore_engine[.\s])", re.MULTILINE
    )
    offenders = []
    for path in sorted((_ROOT / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not engine_import.search(text):
            continue
        if "importorskip" in text or "require_engine" in text:
            continue
        offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == [], f"unguarded module-level engine imports: {offenders}"

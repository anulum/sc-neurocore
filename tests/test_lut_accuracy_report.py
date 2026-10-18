# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the committed lut-v1 accuracy report matches the code

"""``docs/validation/lut_accuracy_v1.md`` is what ``tools/lut_accuracy_report.py`` renders now."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lut_accuracy_report", ROOT / "tools" / "lut_accuracy_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_report_is_current() -> None:
    """The report states exactly the errors the code measures today."""
    assert _tool().main(["--check"]) == 0


def test_report_covers_every_table_and_format() -> None:
    """One row per table function, one column per listed format."""
    tool = _tool()
    text = tool.render()
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert len(rows) == 10
    assert all(line.count("|") == len(tool.FORMATS) + 3 for line in rows)

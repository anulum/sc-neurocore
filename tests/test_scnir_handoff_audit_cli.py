# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cli) from former test_scnir_handoff_audit.py

"""The SC-NIR HDL hand-off audit command-line interface."""

from __future__ import annotations

import json
import pytest
from pathlib import (
    Path,
)
from sc_neurocore.cli import (
    main,
)
from tests.scnir_handoff_audit_support import (
    _write_valid_handoff,
)
from unittest import (
    mock,
)


def test_scnir_audit_hdl_cli_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC-NIR audit HDL CLI writes report."""
    handoff = tmp_path / "handoff"
    report_path = tmp_path / "audit.json"
    _write_valid_handoff(handoff)

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "audit-hdl",
            str(handoff),
            "--output",
            str(report_path),
        ],
    ):
        rc = main()

    assert rc == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "valid"
    assert "SC-NIR HDL handoff valid" in capsys.readouterr().out


def test_scnir_audit_hdl_cli_reports_invalid_handoff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC-NIR audit HDL CLI reports invalid handoff."""
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "scnir_source_manifest.json").unlink()

    with mock.patch("sys.argv", ["sc-neurocore", "scnir", "audit-hdl", str(handoff)]):
        rc = main()

    assert rc == 1
    assert "SC-NIR HDL handoff invalid" in capsys.readouterr().out

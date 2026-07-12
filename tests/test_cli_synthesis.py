# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — synthesis evidence CLI tests

"""Exercise synthesis-evidence dispatch through the public CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cli_test_support import run_cli


def test_collect_synthesis_can_stream_payload_without_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful collector may hand the payload to its default writer destination."""
    import sc_neurocore.optimizer as optimizer

    payload = {"schema_version": "fixture"}
    writes: list[tuple[object, object]] = []
    monkeypatch.setattr(optimizer, "build_payload_from_reports", lambda **_kwargs: payload)
    monkeypatch.setattr(
        optimizer,
        "write_payload",
        lambda value, destination: writes.append((value, destination)),
    )

    assert (
        run_cli(
            "collect-synthesis",
            "--design",
            str(tmp_path / "design.json"),
            "--utilisation",
            str(tmp_path / "utilisation.rpt"),
            "--power",
            str(tmp_path / "power.rpt"),
            "--accuracy-score",
            "0.99",
        )
        == 0
    )
    assert writes == [(payload, None)]
    assert "Evidence written" not in capsys.readouterr().out

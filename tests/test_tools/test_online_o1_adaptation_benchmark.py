# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Online O(1) benchmark CLI

from __future__ import annotations

import json

from tools.online_o1_adaptation_benchmark import main


def test_online_o1_adaptation_benchmark_cli_writes_report(tmp_path, capsys) -> None:
    output = tmp_path / "online_o1_adaptation.json"

    rc = main(["--output", str(output), "--n-synapses", "1024", "--target-weight", "192"])

    assert rc == 0
    assert str(output) in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.online-o1-adaptation-benchmark.v1"
    assert payload["python"]["steps_to_target"] == 16
    assert payload["resource_estimate"]["bram36_tiles"] == 1


def test_online_o1_adaptation_benchmark_cli_rejects_invalid_arguments(capsys) -> None:
    rc = main(["--output", "/tmp/invalid.json", "--n-synapses", "0"])

    assert rc == 1
    assert "online O(1) adaptation benchmark invalid" in capsys.readouterr().err

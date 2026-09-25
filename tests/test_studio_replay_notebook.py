# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A notebook generated from a replay pack cites its model and runs elsewhere

"""The generated notebook's code, run in a fresh process away from the author's files, replays."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.replay_notebook import NOTEBOOK_FORMAT, notebook_from_pack
from sc_neurocore.studio.replay_pack import build_replay_pack

MODEL_REQUEST = {
    "mode": "model",
    "name": "AdExNeuron",
    "current": 500.0,
    "duration": 20.0,
    "dt": 0.1,
}


def _code(notebook: dict[str, Any]) -> str:
    return "\n\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")


def _markdown(notebook: dict[str, Any]) -> str:
    return "\n\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    client = TestClient(create_app(), base_url="http://127.0.0.1")
    response = client.post("/api/export/replay-notebook", json=MODEL_REQUEST)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_notebook_cites_the_model_and_says_what_it_does_not_establish(
    notebook: dict[str, Any],
) -> None:
    assert notebook["nbformat"] == NOTEBOOK_FORMAT
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown",
        "code",
        "code",
        "markdown",
    ]
    text = _markdown(notebook)
    assert (
        "`AdExNeuron` follows Brette, R. & Gerstner, W. (2005) doi:10.1152/jn.00686.2005." in text
    )
    assert "No fixed-point, RTL, synthesis or board step is part of it" in text
    assert notebook["metadata"]["sc_neurocore"]["experiment_sha256"] in text


def test_the_notebook_replays_in_a_fresh_process_without_the_authors_files(
    notebook: dict[str, Any], tmp_path: Path
) -> None:
    script = tmp_path / "notebook.py"
    script.write_text(_code(notebook), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "verdict: match" in run.stdout
    assert "runtime differences: none" in run.stdout


def test_an_equation_experiment_says_it_has_no_published_source() -> None:
    pack = build_replay_pack(
        {
            "mode": "ode",
            "equations": ["dv/dt = -v / 10 + I"],
            "threshold": "v > 1",
            "reset": "v = 0",
            "current": 0.2,
            "duration": 5.0,
            "dt": 0.1,
        }
    )
    assert "no published source is attached" in _markdown(notebook_from_pack(pack))


@pytest.mark.parametrize(
    ("class_name", "phrase"),
    [
        ("NoSuchNeuron", "`NoSuchNeuron` from the SC-NeuroCore catalogue; its descriptor names no"),
        ("ATypeKNeuron", "`ATypeKNeuron` from the SC-NeuroCore catalogue; its descriptor names no"),
    ],
)
def test_a_model_without_a_citeable_descriptor_is_named_as_such(
    class_name: str, phrase: str
) -> None:
    pack = build_replay_pack(MODEL_REQUEST)
    pack["experiment"]["model"]["class_name"] = class_name
    assert phrase in _markdown(notebook_from_pack(pack))


def test_a_cited_paper_title_is_carried_when_the_descriptor_has_one() -> None:
    from sc_neurocore.neurons.model_catalogue import load_descriptor

    descriptor = load_descriptor("AdaptiveThresholdIFNeuron")
    assert descriptor is not None and descriptor.provenance.paper_title
    pack = build_replay_pack(MODEL_REQUEST)
    pack["experiment"]["model"]["class_name"] = "AdaptiveThresholdIFNeuron"
    assert f"*{descriptor.provenance.paper_title}*" in _markdown(notebook_from_pack(pack))

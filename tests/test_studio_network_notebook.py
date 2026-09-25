# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A tutorial notebook rebuilds a drawn network and reproduces its run

"""The notebook's code, run in a fresh process, rebuilds the network the Studio ran."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import create_population, create_projection
from sc_neurocore.studio.network_notebook import (
    NETWORK_NOTEBOOK_KIND,
    _literal,
    spike_events_sha256,
)


def _graph(*, autapses: bool = False) -> dict[str, Any]:
    excitatory = create_population(label="E", count=40, drive={"kind": "constant", "current": 1.2})
    excitatory["id"] = "e"
    inhibitory = create_population(
        label="I",
        count=10,
        neuron_type="inhibitory",
        drive={"kind": "poisson", "rate_hz": 200.0, "weight": 5.0},
    )
    inhibitory["id"] = "i"
    quiet = create_population(label="Q", count=5, model="AdExNeuron")
    quiet["id"] = "q"
    projections = [
        create_projection("e", "i", weight=40.0, delay=0.2, probability=0.3),
        create_projection("i", "e", weight=-40.0, rule="all_to_all"),
        {**create_projection("e", "e", weight=2.0, probability=0.5), "autapses": autapses},
        create_projection("e", "q", weight=30.0, probability=0.2),
    ]
    for projection, name in zip(projections, ("ei", "ie", "ee", "eq"), strict=True):
        projection["id"] = name
    return {
        "populations": [excitatory, inhibitory, quiet],
        "projections": projections,
        "duration": 50.0,
        "dt": 0.1,
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _code(notebook: dict[str, Any]) -> str:
    return "\n\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")


def _markdown(notebook: dict[str, Any]) -> str:
    return "\n\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


def _run(code: str, tmp_path: Path) -> str:
    script = tmp_path / "tutorial.py"
    script.write_text(code, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout


def test_the_notebook_rebuilds_the_network_and_reproduces_every_spike(
    client: TestClient, tmp_path: Path
) -> None:
    graph = _graph()
    response = client.post("/api/graph/notebook", json=graph)
    assert response.status_code == 200, response.text
    notebook = response.json()
    sealed = notebook["metadata"]["sc_neurocore"]
    assert sealed["kind"] == NETWORK_NOTEBOOK_KIND

    # The seal is the Studio's own run of the same graph.
    studio = client.post("/api/graph/simulate", json=graph).json()
    assert sealed["graph_sha256"] == studio["spec"]["graph_sha256"]
    assert sealed["n_spikes"] == studio["n_spikes"] > 0
    assert sealed["spikes_sha256"] == spike_events_sha256(studio)
    assert set(sealed["csr_sha256"]) == {"ei", "ie", "ee", "eq"}

    output = _run(_code(notebook), tmp_path)
    assert f"spikes: {studio['n_spikes']} sealed: {studio['n_spikes']}" in output
    assert "spikes match: True" in output
    for projection in ("ei", "ie", "ee", "eq"):
        assert f"{projection}: " in output
    assert output.count("connectivity match: True") == 4


def test_every_construction_step_is_visible(client: TestClient) -> None:
    notebook = client.post("/api/graph/notebook", json=_graph()).json()
    code = _code(notebook)
    assert "from sc_neurocore.neurons.models.adex import AdExNeuron" in code
    assert "stimulus = StepCurrent(0, N_STEPS, 1.2)" in code
    assert "PoissonInput(10, 200.0, 5.0, dt=DT_S, seed=" in code
    assert "all_to_all(populations['i'].n, populations['e'].n, -40.0)" in code
    # A self-projection without declared autapses drops the diagonal, as the Studio does.
    assert "indptr, indices, data = without_autapses(indptr, indices, data)" in code
    assert "# No external drive: this population is driven only by projections." in code
    assert "= 2 whole steps." in code
    assert 'network.run(duration=N_STEPS * DT_S, dt=DT_S, backend="python")' in code


def test_the_notebook_cites_its_models_and_says_what_it_does_not_establish(
    client: TestClient,
) -> None:
    text = _markdown(client.post("/api/graph/notebook", json=_graph()).json())
    assert (
        "`AdExNeuron` follows Brette, R. & Gerstner, W. (2005) doi:10.1152/jn.00686.2005." in text
    )
    assert "no fixed-point, RTL, synthesis or board step is part of it" in text
    assert "`rust-network-runner` is not used because it constructs populations" in text
    assert "3 populations, 55 neurons, 4 projections, 500 steps of 0.1 ms" in text


def test_declared_autapses_are_kept_and_no_helper_is_written(
    client: TestClient, tmp_path: Path
) -> None:
    notebook = client.post("/api/graph/notebook", json=_graph(autapses=True)).json()
    code = _code(notebook)
    assert "without_autapses" not in code
    assert "spikes match: True" in _run(code, tmp_path)


def test_a_graph_with_no_projections_has_no_projection_section(
    client: TestClient, tmp_path: Path
) -> None:
    graph = _graph()
    graph["projections"] = []
    notebook = client.post("/api/graph/notebook", json=graph).json()
    assert "## Projections" not in _markdown(notebook)
    assert "spikes match: True" in _run(_code(notebook), tmp_path)


def test_a_graph_that_does_not_validate_answers_every_message(client: TestClient) -> None:
    graph = _graph()
    graph["projections"][0]["delay"] = 0.33
    response = client.post("/api/graph/notebook", json=graph)
    assert response.status_code == 422
    assert any(
        "whole number of 0.1 ms steps" in error for error in response.json()["detail"]["errors"]
    )


def test_a_graph_that_fails_while_running_answers_the_failure(client: TestClient) -> None:
    population = create_population(
        count=1, model="AlphaMotorNeuron", drive={"kind": "constant", "current": 1e300}
    )
    population["id"] = "x"
    response = client.post(
        "/api/graph/notebook",
        json={"populations": [population], "projections": [], "duration": 2.0},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "graph_execution_failed"
    assert "non-finite" in detail["reason"]


def test_a_constructor_value_without_an_exact_literal_is_refused() -> None:
    assert (_literal(True), _literal(3), _literal(0.1)) == ("True", "3", "0.1")
    with pytest.raises(TypeError, match="no literal for str"):
        _literal("tau")

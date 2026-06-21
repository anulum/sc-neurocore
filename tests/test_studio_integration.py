# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Integration (Block 6)

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import create_population, create_projection
from sc_neurocore.studio.project import (
    delete_project,
    list_projects,
    load_project,
    run_pipeline,
    save_project,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


# --- Project Save/Load ---


class TestProjectSaveLoad:
    def test_save_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        result = save_project("test_proj", {"dt": 0.1, "duration": 100})
        assert result["name"] == "test_proj"
        assert result["schema_version"] == "studio.project-save.v1"
        assert result["evidence_classification"] == "project_workspace"
        assert result["status"] == "completed"
        assert re.fullmatch(r"[0-9a-f]{64}", result["state_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", result["project_sha256"])
        assert "saved_at" in result
        assert "path" not in result
        assert str(tmp_path) not in json.dumps(result)

    def test_load_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        save_project("load_test", {"equations": ["dv/dt = -v"]})
        result = load_project("load_test")
        assert result["name"] == "load_test"
        assert result["state"]["equations"] == ["dv/dt = -v"]

    def test_load_nonexistent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        result = load_project("nope")
        assert "error" in result

    def test_load_rejects_malformed_project_payload(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        bad_path = tmp_path / "bad_payload.json"
        bad_path.write_text(json.dumps({"name": "bad_payload", "state": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="'state' must be an object"):
            load_project("bad_payload")

    def test_load_rejects_non_object_project_payload(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        bad_path = tmp_path / "bad_payload.json"
        bad_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

        with pytest.raises(ValueError, match="expected object"):
            load_project("bad_payload")

    def test_load_rejects_inconsistent_project_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        bad_path = tmp_path / "expected.json"
        bad_path.write_text(
            json.dumps({"name": "other", "state": {}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="inconsistent project name"):
            load_project("expected")

    def test_list_projects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        save_project("proj_a", {})
        save_project("proj_b", {})
        (tmp_path / "notes.txt").write_text("not a project", encoding="utf-8")
        (tmp_path / "broken.json").write_text("{", encoding="utf-8")
        result = list_projects()
        assert len(result) == 2
        names = {p["name"] for p in result}
        assert "proj_a" in names
        assert "proj_b" in names

    def test_delete_project(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        save_project("to_delete", {})
        assert len(list_projects()) == 1
        result = delete_project("to_delete")
        assert result["deleted"] == "to_delete"
        assert len(list_projects()) == 0

    def test_delete_nonexistent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        result = delete_project("nope")
        assert "error" in result

    def test_save_rejects_non_object_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Project state must be an object"):
            save_project("bad_state", ["not", "an", "object"])  # type: ignore[arg-type]

    def test_save_rejects_invalid_hdl_identifier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Invalid HDL-facing identifiers"):
            save_project("bad_ident", {"module_name": "bad-name"})

    @pytest.mark.parametrize(
        "state",
        [
            {"constants": {"bad-name": 1}},
            {"layers": [{"name": "bad-name"}]},
            {"signals": [{"name": "bad-name"}]},
        ],
    )
    def test_save_rejects_nested_hdl_identifiers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        state: dict[str, Any],
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))

        with pytest.raises(ValueError, match="Invalid HDL-facing identifiers"):
            save_project("bad_nested_ident", state)

    def test_project_names_reject_non_string_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))

        with pytest.raises(ValueError, match="Invalid project name"):
            save_project(123, {})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "..",
            "../escape",
            "..\\escape",
            "/tmp/escape",
            "nested/name",
            "nested\\name",
        ],
    )
    def test_project_names_reject_path_semantics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
    ) -> None:
        monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path))

        with pytest.raises(ValueError, match="Invalid project name"):
            save_project(name, {})
        with pytest.raises(ValueError, match="Invalid project name"):
            load_project(name)
        with pytest.raises(ValueError, match="Invalid project name"):
            delete_project(name)


# --- Pipeline ---


class TestPipeline:
    def _make_graph(self) -> dict[str, object]:
        exc = create_population(count=30, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        return {"populations": [exc, inh], "projections": [proj], "duration": 30.0}

    def test_pipeline_runs(self) -> None:
        graph = self._make_graph()
        result = run_pipeline(graph)
        assert "steps" in result
        assert "validate" in result["steps"]
        assert "simulate" in result["steps"]

    def test_pipeline_empty_graph(self) -> None:
        result = run_pipeline({"populations": [], "projections": []})
        assert result["success"] is False
        assert result["step"] == "validate"

    def test_pipeline_target(self) -> None:
        graph = self._make_graph()
        result = run_pipeline(graph, target="ecp5")
        assert result.get("target") == "ecp5"

    def test_pipeline_reports_simulation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        graph = self._make_graph()

        def fail_simulation(_graph: dict[str, Any]) -> dict[str, object]:
            return {"success": False, "errors": ["sim failed"]}

        monkeypatch.setattr(
            "sc_neurocore.studio.network_graph.simulate_graph",
            fail_simulation,
        )

        result = run_pipeline(graph)

        assert result == {
            "success": False,
            "step": "simulate",
            "errors": ["sim failed"],
        }

    def test_pipeline_reports_compile_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        graph = self._make_graph()

        def fail_compile(*_args: object, **_kwargs: object) -> tuple[object, str]:
            raise RuntimeError("compiler failure")

        monkeypatch.setattr(
            "sc_neurocore.compiler.equation_compiler.equation_to_fpga",
            fail_compile,
        )

        result = run_pipeline(graph)

        assert result == {
            "success": False,
            "step": "compile",
            "error": "Compilation failed",
        }


# --- Endpoints ---


class TestEndpoints:
    def test_save_endpoint(self, client: TestClient) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "endpoint_test", "state": {"dt": 0.1}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "endpoint_test"
        assert data["schema_version"] == "studio.project-save.v1"
        assert data["evidence_classification"] == "project_workspace"
        assert data["status"] == "completed"
        assert re.fullmatch(r"[0-9a-f]{64}", data["state_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", data["project_sha256"])
        assert "path" not in data

    def test_save_requires_name(self, client: TestClient) -> None:
        r = client.post("/api/project/save", json={"state": {}})
        assert r.status_code == 422

    def test_save_rejects_non_object_state_endpoint(
        self,
        client: TestClient,
    ) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "bad_state_endpoint", "state": ["not", "an", "object"]},
        )
        assert r.status_code == 422

    def test_save_rejects_invalid_hdl_identifier_endpoint(
        self,
        client: TestClient,
    ) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "bad_ident_endpoint", "state": {"module_name": "bad-name"}},
        )
        assert r.status_code == 422

    def test_list_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/list")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_load_nonexistent_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/load/nonexistent_xyz")
        assert r.status_code == 404

    def test_load_invalid_name_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/load/bad%5Cname")
        assert r.status_code == 422

    def test_delete_invalid_name_endpoint(self, client: TestClient) -> None:
        r = client.delete("/api/project/bad%5Cname")
        assert r.status_code == 422

    def test_pipeline_endpoint(self, client: TestClient) -> None:
        exc = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=5, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 20.0}
        r = client.post("/api/pipeline/run", json={"graph": graph, "target": "ice40"})
        assert r.status_code == 200
        data = r.json()
        assert "steps" in data

    def test_pipeline_endpoint_empty_graph(self, client: TestClient) -> None:
        r = client.post(
            "/api/pipeline/run",
            json={"graph": {"populations": [], "projections": []}, "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False

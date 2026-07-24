# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio integration project

"""Focused suite: TestProjectSaveLoad from former test_studio_integration.py."""

from __future__ import annotations

from tests.studio_integration_support import *  # noqa: F403


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

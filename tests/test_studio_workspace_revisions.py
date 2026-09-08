# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio workspace revisions and recovery

"""A save must not be able to destroy what was already saved.

The three failures these guard against were each reproduced against the old
single-file store: a second editor's save silently replaced the first, a delete
was unrecoverable, and a write that failed part-way left 65,536 bytes of
unterminated JSON where a 156-byte workspace had been. The write failure here
is a real ``RLIMIT_FSIZE`` refusal in a child process, not a patched function.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from sc_neurocore.studio.workspace_schema import (
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceSchemaError,
    migrate_document,
    read_document,
    write_atomic,
)
from sc_neurocore.studio.workspace_store import WorkspaceConflict, WorkspaceStore

REPO_ROOT = Path(__file__).resolve().parents[1]

FULL_STATE = {
    "experiment": {"name": "HodgkinHuxleyNeuron", "dt": 0.05, "protocol": "step"},
    "graph": {"populations": [{"id": "e", "model": "SCLapicqueLIFNeuron", "count": 4}]},
    "candidates": [{"name": "authored", "equations": ["dv/dt = I"]}],
    "analysis_refs": ["analysis-1"],
    "run_refs": ["sj_0000000000000001"],
    "hardware_profile": {"target": "ice40", "q_format": "Q8.8"},
}


def _store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(root=tmp_path / "workspaces")


class TestRevisions:
    def test_a_save_adds_a_revision_and_never_rewrites_one(self, tmp_path: Path) -> None:
        store = _store(tmp_path)

        first = store.save("w", FULL_STATE, expected_revision=None)
        second = store.save("w", {**FULL_STATE, "note": "second"}, expected_revision=1)

        assert (first.revision, first.parent) == (1, None)
        assert (second.revision, second.parent) == (2, 1)
        assert store.load("w", revision=1)["state"] == FULL_STATE
        assert store.load("w")["state"]["note"] == "second"
        assert [revision.revision for revision in store.revisions("w")] == [1, 2]

    def test_every_state_block_survives_the_round_trip(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", FULL_STATE, expected_revision=None)

        loaded = store.load("w")["state"]

        assert loaded == FULL_STATE
        assert set(loaded) == set(FULL_STATE)

    def test_a_stale_save_is_a_conflict_not_a_lost_update(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"note": "alice"}, expected_revision=None)
        store.save("w", {"note": "bob"}, expected_revision=1)

        with pytest.raises(WorkspaceConflict) as conflict:
            store.save("w", {"note": "alice again"}, expected_revision=1)

        assert conflict.value.expected == 1
        assert conflict.value.actual == 2
        assert conflict.value.to_public_detail()["error"] == "workspace_conflict"
        # Bob's work is still what the workspace holds.
        assert store.load("w")["state"]["note"] == "bob"

    def test_saving_a_new_workspace_over_an_existing_one_is_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"note": "first"}, expected_revision=None)

        with pytest.raises(WorkspaceConflict):
            store.save("w", {"note": "unaware"}, expected_revision=None)

    def test_an_absent_workspace_or_revision_is_absent(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(KeyError):
            store.load("nothing")
        store.save("w", {"note": "one"}, expected_revision=None)
        with pytest.raises(KeyError):
            store.load("w", revision=7)


class TestDurableWrite:
    def test_a_failed_write_leaves_the_previous_revision_intact(self, tmp_path: Path) -> None:
        """A real OS write refusal, in a child process with RLIMIT_FSIZE set."""
        root = tmp_path / "workspaces"
        store = WorkspaceStore(root=root)
        store.save("w", {"note": "the one that matters"}, expected_revision=None)

        child = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "-c",
                "import resource, signal, sys\n"
                "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
                "resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))\n"
                "from pathlib import Path\n"
                "from sc_neurocore.studio.workspace_store import WorkspaceStore\n"
                "store = WorkspaceStore(root=Path(sys.argv[1]))\n"
                "try:\n"
                "    store.save('w', {'blob': 'x' * (4 * 1024 * 1024)}, expected_revision=1)\n"
                "except OSError:\n"
                "    print('refused')\n",
                str(root),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            timeout=600,
            check=False,
        )
        assert "refused" in child.stdout, child.stderr

        # The workspace still reads, still says what it said, and has not moved on.
        assert store.load("w")["state"]["note"] == "the one that matters"
        assert store.head_revision("w") == 1
        assert list((root / "w").glob("**/*.tmp")) == []

    def test_write_atomic_leaves_no_debris_when_the_write_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "file.json"
        target.write_text("original", encoding="utf-8")

        # A payload the encoder refuses: the temporary file is opened and then
        # the write raises, which is the shape of a mid-write failure.
        with pytest.raises(UnicodeEncodeError):
            write_atomic(target, "\udc80")

        assert target.read_text(encoding="utf-8") == "original"
        assert list(tmp_path.glob("*.tmp")) == []


class TestLifecycle:
    def test_same_clock_deletions_remain_separately_restorable(self, tmp_path: Path) -> None:
        """Repeated deletion under one clock tick preserves two independent histories."""
        store = WorkspaceStore(root=tmp_path / "workspaces", clock=lambda: 1.0)
        store.save("w", {"note": "first"})
        first = store.delete("w")
        store.save("w", {"note": "second"})
        second = store.delete("w")
        assert first != second
        assert len(store.deleted()) == 2
        assert all(entry["deleted_at"] == 1.0 for entry in store.deleted())
        store.restore(first.name)
        assert store.load("w")["state"] == {"note": "first"}
        store.delete("w")
        store.restore(second.name)
        assert store.load("w")["state"] == {"note": "second"}

    def test_old_timestamp_only_trash_tokens_remain_restorable(self, tmp_path: Path) -> None:
        """Upgrading the trash-token writer retains existing deletion receipts."""
        store = _store(tmp_path)
        store.save("w", {"note": "before upgrade"})
        deleted = store.delete("w")
        legacy = deleted.with_name("w.1000")
        deleted.rename(legacy)
        assert store.deleted()[0]["deleted_at"] == 1.0
        store.restore(legacy.name)
        assert store.load("w")["state"] == {"note": "before upgrade"}

    def test_a_deleted_workspace_can_be_restored(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"note": "irreplaceable"}, expected_revision=None)
        store.save("w", {"note": "still here"}, expected_revision=1)

        store.delete("w")
        assert store.exists("w") is False
        deleted = store.deleted()
        assert [entry["name"] for entry in deleted] == ["w"]

        assert store.restore(str(deleted[0]["token"])) == "w"
        assert store.head_revision("w") == 2
        assert store.load("w", revision=1)["state"]["note"] == "irreplaceable"

    def test_restoring_onto_a_live_name_is_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"note": "old"}, expected_revision=None)
        store.delete("w")
        store.save("w", {"note": "new work"}, expected_revision=None)

        with pytest.raises(WorkspaceConflict):
            store.restore(str(store.deleted()[0]["token"]))
        # The live workspace is untouched by the refusal.
        assert store.load("w")["state"]["note"] == "new work"

    def test_a_fork_starts_fresh_and_leaves_the_source_alone(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"note": "one"}, expected_revision=None)
        store.save("w", {"note": "two"}, expected_revision=1)

        forked = store.fork("w", "w-fork", revision=1)

        assert forked.revision == 1
        assert store.load("w-fork")["state"]["note"] == "one"
        assert store.head_revision("w") == 2

    def test_forking_onto_an_existing_workspace_is_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("a", {"note": "a"}, expected_revision=None)
        store.save("b", {"note": "b"}, expected_revision=None)

        with pytest.raises(WorkspaceConflict):
            store.fork("a", "b")
        assert store.load("b")["state"]["note"] == "b"

    def test_export_and_import_move_a_workspace_between_installations(self, tmp_path: Path) -> None:
        source = _store(tmp_path / "one")
        source.save("w", FULL_STATE, expected_revision=None)

        document = source.export_document("w")
        assert set(document["state_blocks_present"]) == set(FULL_STATE)

        destination = WorkspaceStore(root=tmp_path / "two" / "workspaces")
        imported = destination.import_document("w", document)

        assert imported.revision == 1
        assert destination.load("w")["state"] == FULL_STATE

    def test_listing_reports_the_current_revision_and_history_depth(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("a", {"n": 1}, expected_revision=None)
        store.save("a", {"n": 2}, expected_revision=1)
        store.save("b", {"n": 1}, expected_revision=None)

        summaries = {entry["name"]: entry for entry in store.list_workspaces()}

        assert summaries["a"]["revision"] == 2
        assert summaries["a"]["revision_count"] == 2
        assert summaries["b"]["revision"] == 1


class TestSchema:
    def test_a_legacy_project_payload_migrates_to_a_revision(self) -> None:
        migrated = migrate_document(
            {"name": "old", "saved_at": 1.0, "version": "0.3.0", "state": {"note": "legacy"}}
        )

        assert migrated["schema_version"] == WORKSPACE_SCHEMA_VERSION
        assert migrated["migrated_from"] == "studio.project-save.v0"
        assert migrated["state"] == {"note": "legacy"}

    def test_a_document_from_a_newer_schema_is_refused(self) -> None:
        with pytest.raises(WorkspaceSchemaError, match="Upgrade the package"):
            migrate_document({"schema_version": "studio.workspace.v99", "state": {}})

    def test_a_document_with_a_newer_schema_number_is_refused(self) -> None:
        with pytest.raises(WorkspaceSchemaError, match="schema number"):
            migrate_document(
                {
                    "schema_version": WORKSPACE_SCHEMA_VERSION,
                    "schema_number": 99,
                    "state": {},
                }
            )

    @pytest.mark.parametrize(
        "document", [[], "text", {"schema_version": WORKSPACE_SCHEMA_VERSION}, {"other": 1}]
    )
    def test_a_document_that_is_not_a_workspace_is_refused(self, document: object) -> None:
        with pytest.raises(WorkspaceSchemaError):
            migrate_document(document)  # type: ignore[arg-type]

    def test_a_corrupt_revision_file_is_reported_without_its_path(self, tmp_path: Path) -> None:
        path = tmp_path / "revision.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(WorkspaceSchemaError) as failure:
            read_document(path)

        assert "not valid JSON" in str(failure.value)
        assert str(tmp_path) not in str(failure.value)

    def test_a_corrupt_revision_is_omitted_from_history_not_hidden(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("w", {"n": 1}, expected_revision=None)
        store.save("w", {"n": 2}, expected_revision=1)
        (tmp_path / "workspaces" / "w" / "revisions" / "1.json").write_text(
            "{corrupt", encoding="utf-8"
        )

        revisions = [revision.revision for revision in store.revisions("w")]

        assert revisions == [2]
        assert store.load("w")["state"]["n"] == 2


class TestLegacyLayout:
    """A workspace saved before revisions existed must not disappear.

    The previous store kept one flat ``<name>.json`` per workspace in the
    project root. A store that only reads ``<name>/head.json`` would find an
    existing installation's saved work simply gone.
    """

    def _legacy(self, tmp_path: Path, name: str, state: dict[str, object]) -> Path:
        root = tmp_path / "workspaces"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.json"
        path.write_text(
            json.dumps({"name": name, "saved_at": 1.5, "version": "0.3.0", "state": state}),
            encoding="utf-8",
        )
        return path

    def test_a_flat_workspace_file_is_adopted_as_revision_one(self, tmp_path: Path) -> None:
        legacy = self._legacy(tmp_path, "old", {"note": "written before revisions"})
        store = _store(tmp_path)

        assert store.load("old")["state"] == {"note": "written before revisions"}
        assert store.head_revision("old") == 1
        assert [entry.revision for entry in store.revisions("old")] == [1]
        # The original file is left where it is: an adoption that writes is
        # reversible, one that deletes is not.
        assert legacy.is_file()

    def test_deleting_an_adopted_workspace_does_not_readopt_it(self, tmp_path: Path) -> None:
        """The retained migration source must not resurrect a deleted older revision."""
        legacy = self._legacy(tmp_path, "old", {"note": "original"})
        original = legacy.read_bytes()
        store = _store(tmp_path)
        store.save("old", {"note": "latest"}, expected_revision=1)
        token = store.delete("old").name
        restarted = _store(tmp_path)
        assert restarted.exists("old") is False
        assert restarted.list_workspaces() == ()
        with pytest.raises(KeyError):
            restarted.load("old")
        assert legacy.read_bytes() == original
        restarted.restore(token)
        assert restarted.load("old")["state"] == {"note": "latest"}
        assert restarted.load("old", revision=1)["state"] == {"note": "original"}

    def test_deleting_a_legacy_workspace_before_first_read_is_recoverable(
        self, tmp_path: Path
    ) -> None:
        """Deletion adopts an unopened flat file before moving its revision to trash."""
        self._legacy(tmp_path, "old", {"note": "never opened"})
        store = _store(tmp_path)
        token = store.delete("old").name
        assert not store.exists("old")
        store.restore(token)
        assert store.load("old")["state"] == {"note": "never opened"}

    def test_an_adopted_workspace_saves_on_top_of_its_own_history(self, tmp_path: Path) -> None:
        self._legacy(tmp_path, "old", {"note": "one"})
        store = _store(tmp_path)

        store.save("old", {"note": "two"}, expected_revision=1)

        assert [entry.revision for entry in store.revisions("old")] == [1, 2]
        assert store.load("old", revision=1)["state"] == {"note": "one"}
        assert store.load("old")["state"] == {"note": "two"}

    def test_saving_over_an_adopted_workspace_without_its_revision_is_refused(
        self, tmp_path: Path
    ) -> None:
        self._legacy(tmp_path, "old", {"note": "one"})
        store = _store(tmp_path)

        with pytest.raises(WorkspaceConflict):
            store.save("old", {"note": "clobber"})

        assert store.load("old")["state"] == {"note": "one"}

    def test_an_adopted_workspace_is_listed(self, tmp_path: Path) -> None:
        self._legacy(tmp_path, "old", {"note": "one"})
        store = _store(tmp_path)

        assert [entry["name"] for entry in store.list_workspaces()] == ["old"]

    def test_a_file_that_is_not_a_workspace_is_left_alone(self, tmp_path: Path) -> None:
        root = tmp_path / "workspaces"
        root.mkdir(parents=True)
        (root / "broken.json").write_text("{", encoding="utf-8")
        (root / "notes.txt").write_text("not a project", encoding="utf-8")
        store = _store(tmp_path)

        # One unreadable file must not take down the listing.
        assert store.list_workspaces() == ()
        assert store.exists("broken") is False
        assert (root / "broken.json").read_text(encoding="utf-8") == "{"

    def test_adoption_is_idempotent(self, tmp_path: Path) -> None:
        self._legacy(tmp_path, "old", {"note": "one"})
        store = _store(tmp_path)

        assert store.adopt_legacy("old") is True
        assert store.adopt_legacy("old") is False
        assert [entry.revision for entry in store.revisions("old")] == [1]


class TestPublicProjectApi:
    def _project_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        import sc_neurocore.studio.project as project

        monkeypatch.setattr(project, "_PROJECTS_DIR", str(tmp_path / "projects"))
        return project

    def test_the_public_api_reports_revisions_and_refuses_a_stale_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)

        first = project.save_project("w", FULL_STATE)
        assert first["revision"] == 1
        assert first["parent_revision"] is None

        project.save_project("w", {"note": "bob"}, expected_revision=1)
        with pytest.raises(WorkspaceConflict):
            project.save_project("w", {"note": "alice"}, expected_revision=1)

        assert project.load_project("w")["state"]["note"] == "bob"
        assert project.load_project("w", revision=1)["state"] == FULL_STATE
        assert [entry["revision"] for entry in project.project_revisions("w")] == [1, 2]

    def test_delete_is_recoverable_through_the_public_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)
        project.save_project("w", {"note": "keep me"})

        deleted = project.delete_project("w")
        assert deleted["recoverable"] is True
        assert "error" in project.load_project("w")

        restored = project.restore_project(project.list_deleted_projects()[0]["token"])
        assert restored["restored"] == "w"
        assert project.load_project("w")["state"]["note"] == "keep me"

    def test_an_unsafe_hdl_identifier_is_still_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="Invalid HDL-facing identifiers"):
            project.save_project("w", {"module_name": "1 bad; name"})

    def test_export_and_import_round_trip_through_the_public_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)
        project.save_project("w", FULL_STATE)

        document = project.export_project("w")
        imported = project.import_project("w-copy", document)

        assert imported["revision"] == 1
        assert project.load_project("w-copy")["state"] == FULL_STATE

    def test_importing_a_document_that_is_not_a_workspace_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)

        result = project.import_project("w", {"nonsense": True})

        assert "Invalid workspace document" in result["error"]

    def test_a_legacy_single_file_project_is_still_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = self._project_module(tmp_path, monkeypatch)
        root = tmp_path / "projects" / "legacy" / "revisions"
        root.mkdir(parents=True)
        (root / "1.json").write_text(
            json.dumps(
                {"name": "legacy", "saved_at": 1.0, "version": "0.3.0", "state": {"note": "old"}}
            ),
            encoding="utf-8",
        )
        (tmp_path / "projects" / "legacy" / "head.json").write_text(
            json.dumps(
                {
                    "schema_version": WORKSPACE_SCHEMA_VERSION,
                    "schema_number": 1,
                    "name": "legacy",
                    "state": {"revision": 1, "saved_at": 1.0},
                }
            ),
            encoding="utf-8",
        )

        loaded = project.load_project("legacy")

        assert loaded["state"] == {"note": "old"}
        assert loaded["migrated_from"] == "studio.project-save.v0"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Two writers of one Studio workspace

"""A conflict check a second writer can walk through is not a check.

The revision store already refused a save made from a stale revision, and the
refusal was real when the writers took turns. They do not take turns. Two
savers read the same head, both computed revision 1, both were acknowledged
with different digests, and one payload was overwritten with nothing to notice
it by — the very loss the store was written to prevent, reappearing in the gap
between reading the head and writing the revision.

These cases hold that gap shut. They drive threads, separate **processes** and
the real HTTP route, because a lock that only excludes threads of one server
protects nobody running two workers; and they check the lock's own promises —
that it is released when the block raises, when the holding process is killed,
and that a writer which cannot take it is refused in bounded time rather than
waiting forever.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.workspace_lock import (
    LOCK_DIR,
    WorkspaceLockTimeout,
    lock_path,
    workspace_lock,
)
from sc_neurocore.studio.workspace_store import (
    WorkspaceConflict,
    WorkspaceRevision,
    WorkspaceStore,
)

STATE = {
    "experiment": {"name": "SCLapicqueLIFNeuron", "dt": 0.1},
    "graph": {"populations": []},
    "hardware_profile": {"target": "ice40"},
}

#: How long one writer waits at the rendezvous for the other to catch up. It
#: only elapses when the store is doing its job and holding the second writer
#: out, so the cases stay deterministic in both worlds.
RENDEZVOUS = 0.75


class _Rendezvous:
    """A clock that lets a second writer catch up, if the store allows it.

    The defect needed both writers inside ``save`` at once. Blocking the clock
    on a barrier put them there: before the fix both passed it and both were
    acknowledged. After the fix the second writer cannot reach the clock at
    all, so the barrier is left to break and the first writer proceeds. The
    same case therefore reproduces the loss on the old code and proves the
    refusal on the new, without either outcome depending on timing luck.
    """

    def __init__(self, parties: int = 2, timeout: float = RENDEZVOUS) -> None:
        self._barrier = threading.Barrier(parties)
        self._timeout = timeout

    def __call__(self) -> float:
        try:
            self._barrier.wait(timeout=self._timeout)
        except threading.BrokenBarrierError:
            pass
        return 1.0


def _run(targets: list[Any]) -> list[Any]:
    """Run callables on their own threads and return their outcomes in order."""
    outcomes: list[Any] = [None] * len(targets)

    def capture(index: int) -> None:
        try:
            outcomes[index] = targets[index]()
        except BaseException as exc:  # noqa: BLE001 - the outcome is the subject
            outcomes[index] = exc

    threads = [threading.Thread(target=capture, args=(index,)) for index in range(len(targets))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "a writer never finished"
    return outcomes


def _sorted_by_type(outcomes: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split outcomes into revisions written and exceptions raised."""
    written = [item for item in outcomes if isinstance(item, WorkspaceRevision)]
    raised = [item for item in outcomes if isinstance(item, BaseException)]
    return written, raised


class TestTheAcknowledgedLostUpdate:
    """The reported defect: two saves acknowledged, one payload kept."""

    def test_two_saves_of_a_new_workspace_acknowledge_exactly_one(self, tmp_path: Path) -> None:
        clock = _Rendezvous()
        stores = [WorkspaceStore(root=tmp_path / "projects", clock=clock) for _ in range(2)]

        outcomes = _run(
            [
                lambda index=index: stores[index].save(
                    "demo", {"editor": index}, expected_revision=None
                )
                for index in range(2)
            ]
        )
        written, raised = _sorted_by_type(outcomes)

        assert len(written) == 1, "two writers were acknowledged for one revision"
        assert [type(error) for error in raised] == [WorkspaceConflict]
        reader = WorkspaceStore(root=tmp_path / "projects")
        assert [revision.revision for revision in reader.revisions("demo")] == [1]
        kept = reader.load("demo")["state"]
        assert kept == {"editor": written[0].name and outcomes.index(written[0])}
        assert written[0].state_sha256 == reader.revisions("demo")[0].state_sha256

    def test_two_saves_from_one_revision_acknowledge_exactly_one(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        WorkspaceStore(root=root).save("demo", {"editor": "first"}, expected_revision=None)
        clock = _Rendezvous()
        stores = [WorkspaceStore(root=root, clock=clock) for _ in range(2)]

        outcomes = _run(
            [
                lambda index=index: stores[index].save(
                    "demo", {"editor": index}, expected_revision=1
                )
                for index in range(2)
            ]
        )
        written, raised = _sorted_by_type(outcomes)

        assert [revision.revision for revision in written] == [2]
        assert [type(error) for error in raised] == [WorkspaceConflict]
        reader = WorkspaceStore(root=root)
        assert [revision.revision for revision in reader.revisions("demo")] == [1, 2]

    def test_the_refused_writer_is_told_which_revision_is_current(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        WorkspaceStore(root=root).save("demo", {"editor": "first"}, expected_revision=None)
        clock = _Rendezvous()
        stores = [WorkspaceStore(root=root, clock=clock) for _ in range(2)]

        outcomes = _run(
            [
                lambda index=index: stores[index].save(
                    "demo", {"editor": index}, expected_revision=1
                )
                for index in range(2)
            ]
        )
        _written, raised = _sorted_by_type(outcomes)
        conflict = raised[0]

        assert isinstance(conflict, WorkspaceConflict)
        assert conflict.expected == 1
        assert conflict.actual == 2
        assert conflict.to_public_detail()["actual_revision"] == 2


_HOLD_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from sc_neurocore.studio.workspace_lock import workspace_lock
    with workspace_lock(Path(sys.argv[1]), sys.argv[2]):
        print("held", flush=True)
        time.sleep(float(sys.argv[3]))
    """
)

_SAVE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from sc_neurocore.studio.workspace_lock import WorkspaceLockTimeout
    from sc_neurocore.studio.workspace_store import WorkspaceConflict, WorkspaceStore
    store = WorkspaceStore(root=Path(sys.argv[1]), lock_timeout=float(sys.argv[3]))
    gate = Path(sys.argv[4])
    while not gate.exists():
        pass
    try:
        revision = store.save("demo", {"editor": sys.argv[2]}, expected_revision=None)
    except WorkspaceConflict as exc:
        print(json.dumps({"outcome": "conflict", "actual": exc.actual}))
    except WorkspaceLockTimeout:
        print(json.dumps({"outcome": "busy"}))
    else:
        print(json.dumps({"outcome": "saved", "revision": revision.revision}))
    """
)


class TestTwoProcesses:
    """A server run with two workers is two processes over one directory."""

    def test_a_second_process_is_refused_while_the_first_holds_the_workspace(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir(parents=True)
        gate = tmp_path / "go"
        gate.touch()
        holder = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _HOLD_SCRIPT, str(root), "demo", "5"],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", _SAVE_SCRIPT, str(root), "second", "0.5", str(gate)],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        finally:
            holder.kill()
            holder.wait(timeout=30)

        assert json.loads(completed.stdout) == {"outcome": "busy"}
        assert not (root / "demo").exists(), "a refused writer must write nothing"

    def test_two_processes_saving_one_new_workspace_leave_one_revision(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "projects"
        root.mkdir(parents=True)
        gate = tmp_path / "go"
        writers = [
            subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", _SAVE_SCRIPT, str(root), name, "30", str(gate)],
                stdout=subprocess.PIPE,
                text=True,
            )
            for name in ("alice", "bob")
        ]
        gate.touch()
        outcomes = []
        for writer in writers:
            stdout, _ = writer.communicate(timeout=90)
            assert writer.returncode == 0, stdout
            outcomes.append(json.loads(stdout))

        assert sorted(outcome["outcome"] for outcome in outcomes) == ["conflict", "saved"]
        reader = WorkspaceStore(root=root)
        assert [revision.revision for revision in reader.revisions("demo")] == [1]
        assert reader.head_revision("demo") == 1
        kept = reader.load("demo")["state"]["editor"]
        assert kept in ("alice", "bob")


class TestTheLockItself:
    """What the exclusion promises beyond keeping two writers apart."""

    def _take_again(self, root: Path) -> str:
        """Take the lock the caller already holds, as a nested operation does."""
        with workspace_lock(root, "demo", timeout=1.0):
            return "taken again"

    def test_one_thread_may_take_it_again(self, tmp_path: Path) -> None:
        with workspace_lock(tmp_path, "demo"):
            assert self._take_again(tmp_path) == "taken again"

    def test_another_thread_is_refused_within_the_wait(self, tmp_path: Path) -> None:
        def attempt() -> str:
            try:
                with workspace_lock(tmp_path, "demo", timeout=0.2):
                    return "acquired"
            except WorkspaceLockTimeout as exc:
                return exc.to_public_detail()["error"]  # type: ignore[return-value]

        with workspace_lock(tmp_path, "demo"):
            assert _run([attempt]) == ["workspace_busy"]

    def test_it_is_released_when_the_block_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="deliberate"), workspace_lock(tmp_path, "demo"):
            raise RuntimeError("deliberate")

        def attempt() -> str:
            with workspace_lock(tmp_path, "demo", timeout=1.0):
                return "acquired"

        assert _run([attempt]) == ["acquired"]

    def test_it_is_released_when_the_holding_process_is_killed(self, tmp_path: Path) -> None:
        holder = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _HOLD_SCRIPT, str(tmp_path), "demo", "60"],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        holder.kill()
        holder.wait(timeout=30)

        with workspace_lock(tmp_path, "demo", timeout=5.0):
            reclaimed = True
        assert reclaimed

    def test_another_process_holding_it_refuses_this_one_in_time(self, tmp_path: Path) -> None:
        holder = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _HOLD_SCRIPT, str(tmp_path), "demo", "10"],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            # Nothing in this process holds the workspace, so the refusal comes
            # from the operating system rather than from an in-process guard.
            with (
                pytest.raises(WorkspaceLockTimeout) as refused,
                workspace_lock(tmp_path, "demo", timeout=0.3),
            ):
                pass
        finally:
            holder.kill()
            holder.wait(timeout=30)

        assert refused.value.name == "demo"

    def test_a_lock_file_that_is_not_a_database_refuses_the_write(self, tmp_path: Path) -> None:
        path = lock_path(tmp_path, "demo")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not a lock")

        # Proceeding without exclusion would be worse than refusing: the whole
        # point of the lock is that a writer never runs unprotected.
        with pytest.raises(sqlite3.DatabaseError), workspace_lock(tmp_path, "demo", timeout=0.3):
            pass

        # The failed attempt released the in-process guard, so the next writer
        # is refused by the same fault rather than by a lock nobody holds.
        path.unlink()
        with workspace_lock(tmp_path, "demo", timeout=1.0):
            recovered = True
        assert recovered

    def test_the_wait_is_reported_without_a_path(self, tmp_path: Path) -> None:
        error = WorkspaceLockTimeout(name="demo", timeout=0.5)
        detail = error.to_public_detail()

        assert detail["error"] == "workspace_busy"
        assert detail["timeout_seconds"] == 0.5
        assert str(tmp_path) not in json.dumps(detail)
        assert "retried" in str(detail["reason"])

    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
    def test_a_name_that_is_not_one_segment_is_refused(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError, match="Invalid workspace name"):
            lock_path(tmp_path, name)

    def test_the_lock_lives_beside_the_workspaces_not_inside_one(self, tmp_path: Path) -> None:
        path = lock_path(tmp_path, "demo")

        assert path.parent == tmp_path / LOCK_DIR
        assert (tmp_path / "demo") not in path.parents

    def test_a_locked_workspace_is_not_listed_as_one(self, tmp_path: Path) -> None:
        store = WorkspaceStore(root=tmp_path / "projects")
        store.save("demo", dict(STATE), expected_revision=None)
        (tmp_path / "projects" / "not-a-workspace").mkdir()

        assert (tmp_path / "projects" / LOCK_DIR).is_dir()
        assert [summary["name"] for summary in store.list_workspaces()] == ["demo"]

    def test_a_directory_named_like_a_legacy_file_is_not_adopted(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root)
        store.save("demo", dict(STATE), expected_revision=None)
        (root / "weird.json").mkdir()

        assert [summary["name"] for summary in store.list_workspaces()] == ["demo"]

    def test_a_workspace_that_was_never_saved_has_no_revisions(self, tmp_path: Path) -> None:
        store = WorkspaceStore(root=tmp_path / "projects")

        assert store.revisions("never-saved") == ()

    def test_deleting_a_workspace_that_is_not_there_releases_the_lock(self, tmp_path: Path) -> None:
        store = WorkspaceStore(root=tmp_path / "projects")

        with pytest.raises(KeyError):
            store.delete("never-saved")

        # The refusal must not leave the name held: saving it now has to work.
        assert store.save("never-saved", dict(STATE), expected_revision=None).revision == 1


class TestAnInterruptedWriter:
    """A writer that died between its two writes must cost nothing."""

    def test_a_revision_the_head_does_not_point_at_is_never_overwritten(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root)
        store.save("demo", {"editor": "first"}, expected_revision=None)
        orphan = root / "demo" / "revisions" / "2.json"
        orphan.write_text(
            json.dumps(
                {
                    "schema_version": "studio.workspace.v1",
                    "name": "demo",
                    "revision": 2,
                    "parent": 1,
                    "saved_at": 2.0,
                    "version": "0.3.0",
                    "state": {"editor": "interrupted"},
                }
            ),
            encoding="utf-8",
        )

        written = store.save("demo", {"editor": "third"}, expected_revision=1)

        assert written.revision == 3
        assert json.loads(orphan.read_text(encoding="utf-8"))["state"] == {"editor": "interrupted"}
        assert [revision.revision for revision in store.revisions("demo")] == [1, 2, 3]

    def test_the_interrupted_state_is_still_readable(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root)
        store.save("demo", {"editor": "first"}, expected_revision=None)
        (root / "demo" / "revisions" / "2.json").write_text(
            json.dumps(
                {
                    "schema_version": "studio.workspace.v1",
                    "name": "demo",
                    "revision": 2,
                    "parent": 1,
                    "saved_at": 2.0,
                    "version": "0.3.0",
                    "state": {"editor": "interrupted"},
                }
            ),
            encoding="utf-8",
        )

        assert store.load("demo", revision=2)["state"] == {"editor": "interrupted"}


class TestLifecycleUnderTheLock:
    """Deleting and restoring move a whole workspace; they race a save too."""

    def test_a_delete_waits_for_the_writer_that_holds_the_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root, lock_timeout=0.3)
        store.save("demo", dict(STATE), expected_revision=None)

        def delete() -> str:
            try:
                store.delete("demo")
            except WorkspaceLockTimeout:
                return "busy"
            return "deleted"

        with workspace_lock(root, "demo"):
            assert _run([delete]) == ["busy"]
        assert (root / "demo").is_dir()

    def test_nothing_is_waiting_before_anything_is_deleted(self, tmp_path: Path) -> None:
        store = WorkspaceStore(root=tmp_path / "projects")
        store.save("demo", dict(STATE), expected_revision=None)

        assert store.deleted() == ()

    def test_a_stray_file_in_the_trash_is_not_a_deleted_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root)
        store.save("demo", dict(STATE), expected_revision=None)
        token = store.delete("demo").name
        (root / ".trash" / "note.txt").write_text("not a workspace", encoding="utf-8")

        assert [entry["token"] for entry in store.deleted()] == [token]

    def test_two_restores_of_one_name_leave_one_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        store = WorkspaceStore(root=root)
        tokens = []
        for index in range(2):
            store.save("demo", {"editor": index}, expected_revision=None)
            tokens.append(store.delete("demo").name)

        outcomes = _run([lambda token=token: store.restore(token) for token in tokens])
        restored = [item for item in outcomes if item == "demo"]
        refused = [item for item in outcomes if isinstance(item, BaseException)]

        assert len(restored) == 1
        assert [type(error) for error in refused] == [WorkspaceConflict]
        assert (root / "demo" / "head.json").is_file()
        assert not (root / "demo" / "demo").exists(), "a restore moved into the other's directory"

    def test_a_fork_and_a_save_of_one_destination_acknowledge_exactly_one(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "projects"
        WorkspaceStore(root=root).save("source", {"editor": "source"}, expected_revision=None)
        clock = _Rendezvous()
        forker = WorkspaceStore(root=root, clock=clock)
        saver = WorkspaceStore(root=root, clock=clock)

        outcomes = _run(
            [
                lambda: forker.fork("source", "target"),
                lambda: saver.save("target", {"editor": "direct"}, expected_revision=None),
            ]
        )
        written, raised = _sorted_by_type(outcomes)

        assert len(written) == 1
        assert [type(error) for error in raised] == [WorkspaceConflict]
        assert [
            revision.revision for revision in WorkspaceStore(root=root).revisions("target")
        ] == [1]


class TestAWorkspaceSavedBeforeRevisions:
    """An existing installation's flat files still adopt exactly once."""

    def _legacy(self, root: Path, name: str, state: dict[str, object]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.json"
        path.write_text(
            json.dumps({"name": name, "saved_at": 1.5, "version": "0.3.0", "state": state}),
            encoding="utf-8",
        )
        return path

    def test_two_readers_adopt_one_flat_file_once(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        self._legacy(root, "old", {"note": "written before revisions"})
        stores = [WorkspaceStore(root=root) for _ in range(2)]

        outcomes = _run([lambda index=index: stores[index].exists("old") for index in range(2)])

        assert outcomes == [True, True]
        assert [revision.revision for revision in stores[0].revisions("old")] == [1]
        assert stores[0].load("old")["state"] == {"note": "written before revisions"}

    def test_a_save_over_an_adopted_workspace_states_its_revision(self, tmp_path: Path) -> None:
        root = tmp_path / "projects"
        self._legacy(root, "old", {"note": "written before revisions"})
        store = WorkspaceStore(root=root)

        with pytest.raises(WorkspaceConflict):
            store.save("old", {"note": "blind"}, expected_revision=None)
        written = store.save("old", {"note": "informed"}, expected_revision=1)

        assert written.revision == 2


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A Studio client whose workspaces live under this test's own directory."""
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr("sc_neurocore.studio.project._LOCK_TIMEOUT", 0.3)
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


class TestOverHttp:
    """The browser is the editor, so the guarantee has to reach it."""

    def test_two_clients_saving_one_new_workspace_get_one_success(self, client: TestClient) -> None:
        def save(editor: str) -> tuple[int, Any]:
            response = client.post(
                "/api/project/save",
                json={"name": "shared", "state": {**STATE, "editor": editor}},
            )
            return response.status_code, response.json()

        outcomes = _run([lambda: save("alice"), lambda: save("bob")])
        statuses = sorted(status for status, _ in outcomes)

        assert statuses == [200, 409]
        loaded = client.get("/api/project/load/shared")
        assert loaded.status_code == 200
        assert loaded.json()["state"]["editor"] in ("alice", "bob")

    def test_a_busy_workspace_answers_503_without_a_path(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        root = (tmp_path / "projects").resolve()
        root.mkdir(parents=True, exist_ok=True)

        def hold() -> None:
            barrier.wait(timeout=10)
            with workspace_lock(root, "busy", timeout=5.0):
                barrier.wait(timeout=10)
                barrier.wait(timeout=10)

        barrier = threading.Barrier(2)
        holder = threading.Thread(target=hold)
        holder.start()
        barrier.wait(timeout=10)
        barrier.wait(timeout=10)
        try:
            response = client.post("/api/project/save", json={"name": "busy", "state": dict(STATE)})
        finally:
            barrier.wait(timeout=10)
            holder.join(timeout=30)

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["error"] == "workspace_busy"
        assert str(root) not in response.text
        assert not (root / "busy").exists()

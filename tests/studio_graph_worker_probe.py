# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kernel observations at an actual graph worker loop barrier

"""Stop a real graph worker inside its numerical loop without replacing execution."""

from __future__ import annotations

import json
import os
import resource
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class GraphWorkerProbe:
    """Test-owned startup instrumentation and its actual worker observation."""

    directory: Path
    marker: Path

    def stopped_pid(self) -> int:
        """Wait for an actual in-loop observation followed by a kernel-stopped process."""
        deadline = time.monotonic() + 30.0
        while not self.marker.exists():
            assert time.monotonic() < deadline, "Graph worker never reached its loop barrier"
            time.sleep(0.01)
        record = json.loads(self.marker.read_text())
        assert record["task"] == "run_graph_job" and record["step"] == 1
        pid = int(record["pid"])
        while "\nState:\tT" not in Path(f"/proc/{pid}/status").read_text():
            assert time.monotonic() < deadline, "Graph worker did not stop at its barrier"
            time.sleep(0.01)
        assert Path(f"/proc/{pid}/stat").read_text().split()[21] == record["start_ticks"]
        return pid

    def assert_limits(self, expected: dict[int, tuple[int, int]]) -> None:
        """Read the kernel limits of that exact stopped graph worker, not a generic child."""
        pid = self.stopped_pid()
        for kind, limits in expected.items():
            assert resource.prlimit(pid, kind) == limits


@pytest.fixture
def graph_worker_probe(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[GraphWorkerProbe | None]:
    """Install opt-in independent Python monitoring for the real graph worker only."""
    if not getattr(request, "param", False):
        yield None
        return
    directory = tmp_path / "graph-worker-observer"
    directory.mkdir()
    marker = directory / "loop.json"
    code = f"""import json, os, signal, sys
from pathlib import Path

def observe(frame, event, argument):
    if event != "call":
        return
    loop = frame.f_back
    if loop is None or loop.f_code.co_name != "_run_python" or not loop.f_code.co_filename.endswith("/sc_neurocore/network/network.py"):
        return
    if loop.f_locals.get("step") != 1:
        return
    caller = loop
    while caller is not None and caller.f_code.co_name != "run_graph_job":
        caller = caller.f_back
    if caller is None:
        return
    sys.setprofile(None)
    Path({str(marker)!r}).write_text(json.dumps({{"pid": os.getpid(), "start_ticks": Path("/proc/self/stat").read_text().split()[21], "task": caller.f_code.co_name, "step": loop.f_locals["step"]}}))
    os.kill(os.getpid(), signal.SIGSTOP)

assert sys.getprofile() is None, "A graph worker already has a profile observer"
sys.setprofile(observe)
"""
    header = "\n".join(Path(__file__).read_text().splitlines()[:7]) + "\n"
    (directory / "sitecustomize.py").write_text(header + code)
    monkeypatch.setenv("PYTHONPATH", str(directory) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        yield GraphWorkerProbe(directory, marker)
    finally:
        if marker.exists():
            observed = json.loads(marker.read_text())
            pid = int(observed["pid"])
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().split()
                if stat[21] == observed["start_ticks"] and stat[2] == "T":
                    os.kill(pid, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                pass

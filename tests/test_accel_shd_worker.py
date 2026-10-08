# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated SHD native-result and lifetime contracts

"""Exercise native result admission and watchdog exits through real SHD processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.accel.shd_recordings import read_shd_recording
from tests.test_accel_go_shd_abi import recording_file as recording_file
from tests.test_accel_go_shd_abi import shd_library as shd_library


@pytest.fixture(scope="module", params=["unaligned", "over_budget", "null_buffer"])
def damaged_shd_library(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Compile the real Go reader with a damaged C result and observable release."""
    root = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go"
    source = root / "services/loaders/cshared"
    original = (source / "shd.go").read_text()
    if request.param == "unaligned":
        damaged = original.replace("C.size_t(len(sample.Events))", "C.size_t(len(sample.Events)+1)")
    elif request.param == "over_budget":
        damaged = original.replace("C.size_t(len(sample.Events))", "C.size_t(len(sample.Events)+4)")
    else:
        damaged = original.replace(
            "recording.events = (*C.double)(allocation)",
            "C.free(allocation)\n\trecording.events = nil",
        )
    assert damaged != original
    damaged = damaged.replace('"unsafe"', '"unsafe"\n\t"os"')
    damaged = damaged.replace(
        "\trecording.label = 0\n}",
        '\trecording.label = 0\n\tif err := os.WriteFile(os.Getenv("SHD_RELEASE_PROBE"), '
        '[]byte("released"), 0600); err != nil { panic(err) }\n}',
    )
    directory = tmp_path_factory.mktemp(f"shd-result-{request.param}")
    (directory / "main.go").write_bytes((source / "main.go").read_bytes())
    (directory / "shd.go").write_text(damaged)
    library = directory / "libshd.so"
    subprocess.run(
        [
            "go",
            "build",
            "-tags=hdf5",
            "-buildmode=c-shared",
            "-o",
            str(library),
            str(directory / "main.go"),
            str(directory / "shd.go"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return library


def test_public_reader_refuses_damaged_native_result_and_releases_ownership(
    tmp_path: Path,
    recording_file: Path,
    damaged_shd_library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed native metadata never publishes values and still calls the real free export."""
    release = tmp_path / "released.txt"
    monkeypatch.setenv("SC_NEUROCORE_SHD_GO_LIBRARY", str(damaged_shd_library))
    monkeypatch.setenv("SHD_RELEASE_PROBE", str(release))
    with pytest.raises(RuntimeError, match="Go SHD reader failed or refused the recording"):
        read_shd_recording(recording_file, 0, maximum_bytes=64, backend="go")
    assert release.read_text() == "released"


@pytest.mark.parametrize("parent_matches", [True, False])
def test_worker_watchdog_terminates_and_reaps_a_blocked_native_read(
    tmp_path: Path, shd_library: Path, parent_matches: bool
) -> None:
    """A real FIFO read ends at the actual lifetime or when the supplied parent is different."""
    fifo = tmp_path / "blocked.h5"
    os.mkfifo(fifo)
    worker = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/shd_worker.py"
    parent = os.getpid() if parent_matches else os.getppid()
    result = subprocess.run(
        [sys.executable, "-I", str(worker), str(shd_library), str(fifo), "0", "64", str(parent)],
        capture_output=True,
        timeout=40,
    )
    assert result.returncode == 124
    assert result.stdout == b""


def test_blocked_native_worker_exits_when_its_real_parent_dies(
    tmp_path: Path, shd_library: Path
) -> None:
    """A contained subreaper kills the actual caller and reaps its blocked SHD worker."""
    fifo = tmp_path / "orphaned.h5"
    os.mkfifo(fifo)
    worker = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/shd_worker.py"
    probe = r'''
import ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(36, 1, 0, 0, 0) == 0, ctypes.get_errno()
ready = Path(sys.argv[4])
caller = r"""
import os, subprocess, sys
from pathlib import Path
worker = subprocess.Popen(
    [sys.executable, '-I', sys.argv[1], sys.argv[2], sys.argv[3], '0', '64', str(os.getpid())],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
Path(sys.argv[4]).write_text(str(worker.pid))
worker.wait()
"""
parent = subprocess.Popen(
    [sys.executable, '-I', '-c', caller, *sys.argv[1:]],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
)
try:
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    reader = int(ready.read_text())
    channel = Path(f'/proc/{reader}/wchan')
    while channel.read_text().strip() != 'wait_for_partner' and time.monotonic() < deadline:
        time.sleep(0.01)
    assert channel.read_text().strip() == 'wait_for_partner'
    parent.kill()
    parent.communicate(timeout=5)
    assert parent.returncode == -signal.SIGKILL
    deadline = time.monotonic() + 5
    waited, status = os.waitpid(reader, os.WNOHANG)
    while not waited and time.monotonic() < deadline:
        time.sleep(0.01)
        waited, status = os.waitpid(reader, os.WNOHANG)
    assert waited == reader
    assert os.waitstatus_to_exitcode(status) == 124
    assert not Path(f'/proc/{reader}').exists()
    print(json.dumps({'parent_killed': parent.pid, 'worker_reaped': reader, 'exit': 124}))
finally:
    try:
        os.killpg(parent.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    parent.communicate(timeout=5)
'''
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            probe,
            str(worker),
            str(shd_library),
            str(fifo),
            str(tmp_path / "worker.pid"),
        ],
        capture_output=True,
        check=True,
        timeout=25,
    )
    receipt = json.loads(result.stdout)
    assert receipt["exit"] == 124
    assert receipt["parent_killed"] != receipt["worker_reaped"]

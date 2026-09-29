# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native DVS command protocol and interruption

"""Qualify public reader refusal using real compiled command mutations and actual signals."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_go_dvs_recordings import go_dvs_executable as go_dvs_executable


@pytest.fixture(scope="module", params=["short", "magic", "count", "budget", "payload", "trailing"])
def malformed_go_command(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Compile actual native reader commands whose successful transport output is damaged."""
    source = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go"
    original = (source / "services/loaders/dvscli/main.go").read_text()
    mutation = str(request.param)
    if mutation == "short":
        text = original.replace('[]byte("DVS1")', '[]byte("DVS")').replace(
            "err = binary.Write(os.Stdout, binary.LittleEndian, uint64(len(events)))",
            "os.Exit(0)",
        )
    elif mutation == "magic":
        text = original.replace('[]byte("DVS1")', '[]byte("FAIL")')
    elif mutation == "count":
        text = original.replace("uint64(len(events))", "uint64(len(events)+1)")
    elif mutation == "budget":
        text = original.replace("uint64(len(events))", "uint64(len(events)+4)")
    elif mutation == "payload":
        text = original.replace(
            "binary.LittleEndian, events)", "binary.LittleEndian, events[:len(events)-1])"
        )
    else:
        text = original.replace(
            "binary.LittleEndian, events)", "binary.LittleEndian, append(events, 1))"
        )
    assert text != original
    directory = tmp_path_factory.mktemp(f"go-protocol-{mutation}")
    main = directory / "main.go"
    main.write_text(text)
    lifetime = directory / "lifetime_linux.go"
    lifetime.write_bytes((source / "services/loaders/dvscli/lifetime_linux.go").read_bytes())
    executable = directory / "dvs-go"
    subprocess.run(
        ["go", "build", "-o", str(executable), str(main), str(lifetime)],
        cwd=source,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return executable


def test_public_reader_refuses_actual_native_transport_mutation(
    tmp_path: Path, malformed_go_command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a successful native decode/status cannot admit an incomplete or invalid event frame."""
    recording = tmp_path / "events.npy"
    np.save(recording, np.zeros((2, 4)))
    command = subprocess.run(
        [str(malformed_go_command), str(recording), "64", str(os.getpid())],
        capture_output=True,
        check=True,
        timeout=5,
    )
    assert command.returncode == 0
    monkeypatch.setenv("SC_NEUROCORE_DVS_GO_EXE", str(malformed_go_command))
    with pytest.raises(RuntimeError, match="incomplete response|invalid response"):
        read_dvs_recording(recording, maximum_bytes=64)


def test_invalid_public_backend_refuses_before_file_io(tmp_path: Path) -> None:
    """A dynamically supplied unsupported selector is refused without touching a missing file."""
    with pytest.raises(ValueError, match="unsupported.*backend"):
        read_dvs_recording(tmp_path / "missing.npy", backend=json.loads('"invalid"'))


def test_actual_interruption_reaps_the_blocked_native_command(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """A contained Python caller receives a real signal and proves its native child was reaped."""
    fifo = tmp_path / "blocked.npy"
    os.mkfifo(fifo)
    probe = """
import json, os, signal, sys
from pathlib import Path
if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()
sys.path.insert(0, sys.argv[2])
from sc_neurocore.accel import dvs_recordings
assert Path(dvs_recordings.__file__).is_relative_to(Path(sys.argv[2]))
read_dvs_recording = dvs_recordings.read_dvs_recording
children = Path(f'/proc/self/task/{os.getpid()}/children')
before = set(children.read_text().split())
reader_pid = None

def interrupt(signum, frame):
    global reader_pid
    current = set(children.read_text().split()) - before
    assert len(current) == 1, current
    reader_pid = int(current.pop())
    assert Path(f'/proc/{reader_pid}/wchan').read_text().strip() == 'wait_for_partner'
    raise KeyboardInterrupt('actual native DVS caller interruption')

signal.signal(signal.SIGALRM, interrupt)
signal.alarm(1)
try:
    read_dvs_recording(Path(sys.argv[1]))
except KeyboardInterrupt:
    assert reader_pid is not None
    assert set(children.read_text().split()) == before
    assert not Path(f'/proc/{reader_pid}').exists()
    print(json.dumps({'interrupted': True, 'reaped': reader_pid}))
else:
    raise AssertionError('blocked native read returned without interruption')
finally:
    signal.alarm(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(fifo), str(Path(__file__).resolve().parents[1] / "src")],
        env={**os.environ, "SC_NEUROCORE_DVS_GO_EXE": str(go_dvs_executable)},
        capture_output=True,
        check=True,
        timeout=10,
    )
    receipt = json.loads(result.stdout)
    assert receipt["interrupted"] is True and receipt["reaped"] > 0


@pytest.fixture(scope="module")
def slow_go_command(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the real reader with a longer child deadline to qualify the parent's independent bound."""
    source = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go"
    original = (source / "services/loaders/dvscli/main.go").read_text()
    text = original.replace("30*time.Second", "60*time.Second")
    assert text != original
    directory = tmp_path_factory.mktemp("go-long-child-deadline")
    main = directory / "main.go"
    main.write_text(text)
    lifetime = directory / "lifetime_linux.go"
    lifetime.write_bytes((source / "services/loaders/dvscli/lifetime_linux.go").read_bytes())
    executable = directory / "dvs-go"
    subprocess.run(
        ["go", "build", "-o", str(executable), str(main), str(lifetime)],
        cwd=source,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return executable


def test_parent_timeout_kills_and_reaps_a_longer_lived_real_native_reader(
    tmp_path: Path, slow_go_command: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public parent's 30-second wait remains bounded even when the declared child's timer is longer."""
    fifo = tmp_path / "blocked.npy"
    os.mkfifo(fifo)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    monkeypatch.setenv("SC_NEUROCORE_DVS_GO_EXE", str(slow_go_command))
    with pytest.raises(RuntimeError, match="exceeded its 30 second lifetime"):
        read_dvs_recording(fifo)
    assert set(children.read_text().split()) == before

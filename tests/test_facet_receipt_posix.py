# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recorder process groups after their leader exits

"""Check real Linux process-group cleanup with an already exited group leader."""

from __future__ import annotations

import _thread
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.facet_receipt_budget_support import command_budget, poll_interval
from tools.facet_receipt import run_command


@contextmanager
def _adopt_exited_child_descendants() -> Iterator[None]:
    """Reap this test's orphaned descendant and restore the prior subreaper setting."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    libc.prctl.restype = ctypes.c_int
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        yield
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_linux_interruption_kills_the_remaining_group_after_its_leader_exits(
    tmp_path: Path,
) -> None:
    """A real leader exit cannot leave its output-holding descendant running after interruption."""
    assert sys.platform.startswith("linux")
    started = time.monotonic()
    status, counts, outcome, _ = run_command([sys.executable, "-c", "pass"], timeout=None)
    baseline = time.monotonic() - started
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    budget = command_budget(baseline)
    interval = poll_interval(baseline)
    identity = tmp_path / "leader-and-child.json"
    completed = tmp_path / "grandchild-completed"
    child = (
        "import sys,time;from pathlib import Path;time.sleep(float(sys.argv[2]));"
        "Path(sys.argv[1]).write_text('completed')"
    )
    leader = (
        "import subprocess,os,json,sys;from pathlib import Path;"
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2],sys.argv[4]]);"
        "Path(sys.argv[1]).write_text(json.dumps({'leader':os.getpid(),'child':p.pid}));"
        "print('leader exiting',flush=True)"
    )
    command = [
        sys.executable,
        "-c",
        leader,
        str(identity),
        str(completed),
        child,
        str(2 * budget),
    ]
    stopped = threading.Event()
    observed = threading.Event()
    seen: dict[str, int] = {}
    deadline = time.monotonic() + budget

    def interrupt_after_leader_exit() -> None:
        """Interrupt only after the actual leader is exited and its child group is observed."""
        while not stopped.wait(interval) and time.monotonic() < deadline:
            if not identity.is_file():
                continue
            try:
                facts = json.loads(identity.read_text())
                leader_pid = int(facts["leader"])
                child_pid = int(facts["child"])
                state_path = Path(f"/proc/{leader_pid}/stat")
                if not state_path.exists() or state_path.read_text().split()[2] == "Z":
                    seen.update(leader=leader_pid, child=child_pid, pgid=os.getpgid(child_pid))
                    observed.set()
                    _thread.interrupt_main()
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                continue

    control = threading.Thread(target=interrupt_after_leader_exit)
    reaped: tuple[int, int] | None = None
    with _adopt_exited_child_descendants():
        try:
            control.start()
            with pytest.raises(KeyboardInterrupt):
                run_command(command, timeout=budget)
        finally:
            stopped.set()
            control.join(timeout=budget)
            assert not control.is_alive()
            if identity.is_file():
                facts = json.loads(identity.read_text())
                reaped = os.waitpid(int(facts["child"]), 0)
    assert observed.is_set()
    assert seen["pgid"] == seen["leader"]
    assert reaped is not None
    assert reaped[0] == seen["child"]
    assert os.waitstatus_to_exitcode(reaped[1]) == -signal.SIGKILL
    assert not completed.exists()
    for pid in (seen["leader"], seen["child"]):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    (tmp_path / "group-interruption-control.json").write_text(
        json.dumps(
            {
                "observed": seen,
                "reaped": reaped,
                "baseline_seconds": baseline,
                "budget_seconds": budget,
                "poll_interval_seconds": interval,
                "command": command,
                "scientific_receipt_credit": False,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("leader_finishes", "child_detaches"), [(False, True), (True, True), (True, False)]
)
def test_linux_detached_output_writer_cannot_extend_the_owned_command_deadline(
    tmp_path: Path,
    leader_finishes: bool,
    child_detaches: bool,
) -> None:
    """Respect the actual owned scope while descendants retain inherited output handles."""
    assert sys.platform.startswith("linux")
    started = time.monotonic()
    status, counts, outcome, _ = run_command([sys.executable, "-c", "pass"], timeout=None)
    baseline = time.monotonic() - started
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    budget = command_budget(baseline, ceiling=4 * 60 * 60 / 8)
    identity = tmp_path / "leader-detached-child.json"
    completed = tmp_path / "detached-child-completed"
    child = (
        "import sys,time;from pathlib import Path;time.sleep(float(sys.argv[2]));"
        "Path(sys.argv[1]).write_text('completed')"
    )
    leader = (
        "import subprocess,os,json,sys,time;from pathlib import Path;"
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[3],sys.argv[2],sys.argv[4]],"
        "start_new_session=sys.argv[6]=='1');"
        "Path(sys.argv[1]).write_text(json.dumps({'leader':os.getpid(),'child':p.pid,"
        "'child_pgid':os.getpgid(p.pid)}));"
        "os.write(1,b'leader ready\\n');os.write(2,b'stderr ready\\n');"
        "time.sleep(0 if sys.argv[5]=='1' else float(sys.argv[4]))"
    )
    command = [
        sys.executable,
        "-c",
        leader,
        str(identity),
        str(completed),
        child,
        str(4 * budget),
        str(int(leader_finishes)),
        str(int(child_detaches)),
    ]
    facts: dict[str, int] = {}
    reaped: tuple[int, int] | None = None
    with _adopt_exited_child_descendants():
        try:
            started = time.monotonic()
            status, counts, outcome, tool = run_command(command, timeout=budget)
            elapsed = time.monotonic() - started
            facts = json.loads(identity.read_text())
            assert facts["child_pgid"] == facts["child" if child_detaches else "leader"]
            timed_out = not (leader_finishes and child_detaches)
            assert status == (-1 if timed_out else 0) and counts["passed"] == 0
            assert outcome == ("timeout" if timed_out else "skipped")
            assert tool["stdout"] == "leader ready\n"
            assert tool["stderr"] == "stderr ready\n"
            assert not completed.exists()
            assert elapsed < 2 * budget
            reaped = os.waitpid(facts["child"], os.WNOHANG if child_detaches else 0)
            if child_detaches:
                assert reaped == (0, 0)
            else:
                assert os.waitstatus_to_exitcode(reaped[1]) == -signal.SIGKILL
        finally:
            if identity.is_file():
                facts = json.loads(identity.read_text())
                if reaped is None:
                    reaped = os.waitpid(facts["child"], os.WNOHANG)
                if reaped == (0, 0):
                    os.kill(facts["child"], signal.SIGKILL)
                    reaped = os.waitpid(facts["child"], 0)
    assert reaped is not None and reaped[0] == facts["child"]
    assert os.waitstatus_to_exitcode(reaped[1]) == -signal.SIGKILL
    for pid in (facts["leader"], facts["child"]):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    (tmp_path / "detached-output-control.json").write_text(
        json.dumps(
            {
                "observed": facts,
                "reaped": reaped,
                "baseline_seconds": baseline,
                "budget_seconds": budget,
                "elapsed_seconds": elapsed,
                "command": command,
                "scientific_receipt_credit": False,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("protocol", ["racing", "coordinated"])
def test_linux_concurrent_output_truncation_cannot_crash_the_public_recorder(
    tmp_path: Path,
    protocol: str,
) -> None:
    """Preserve resize stress and truncate again after a real kernel read notification."""
    assert sys.platform.startswith("linux")
    started = time.monotonic()
    status, counts, outcome, _ = run_command(
        [sys.executable, "-c", "from tools.facet_receipt import run_command"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    budget = command_budget(baseline, ceiling=4 * 60 * 60 / 8)
    script = tmp_path / "concurrent_writer.py"
    repo_root = Path(__file__).resolve().parents[1]
    script.write_text(
        "import ctypes,json,os,resource,select,signal,struct,subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"sys.path[:0]=[{str(repo_root)!r},{str(repo_root / 'src')!r}]\n"
        "resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
        "mode,identity,guardian,extent,budget,protocol=sys.argv[1:7]\n"
        "identity=Path(identity);guardian=int(guardian);extent=int(extent);budget=float(budget)\n"
        "recorder=int(sys.argv[7]) if len(sys.argv)>7 else os.getpid()\n"
        "if mode=='writer' and protocol=='coordinated':\n"
        " deadline=time.monotonic()+budget;libc=ctypes.CDLL(None,use_errno=True)\n"
        " start=Path(f'/proc/{recorder}/stat').read_text().rsplit(')',1)[1].split()[19]\n"
        " def signal_recorder(sig):\n"
        "  assert Path(f'/proc/{recorder}/stat').read_text().rsplit(')',1)[1].split()[19]==start\n"
        "  os.kill(recorder,sig)\n"
        " def stop_recorder():\n"
        "  signal_recorder(signal.SIGSTOP)\n"
        "  while time.monotonic()<deadline:\n"
        "   if Path(f'/proc/{recorder}/stat').read_text().rsplit(')',1)[1].split()[0]=='T':return\n"
        "   time.sleep(0.001)\n"
        "  raise TimeoutError('Owned recorder did not stop')\n"
        " def read_count():\n"
        "  return int(dict(line.split(':',1) for line in Path(f'/proc/{recorder}/io').read_text().splitlines())['rchar'])\n"
        " notify=libc.inotify_init1(os.O_CLOEXEC);assert notify>=0\n"
        " libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]\n"
        " assert libc.inotify_add_watch(notify,b'/proc/self/fd/1',1)>=0\n"
        " os.ftruncate(1,extent);stop_recorder()\n"
        " identity.write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))\n"
        " while os.getppid()!=guardian and time.monotonic()<deadline:time.sleep(0.001)\n"
        " assert os.getppid()==guardian and libc.prctl(1,signal.SIGKILL,0,0,0)==0\n"
        " assert os.getppid()==guardian\n"
        " before=read_count();signal_recorder(signal.SIGCONT)\n"
        " ready,_,_=select.select([notify],[],[],max(0,deadline-time.monotonic()));assert ready\n"
        " event=os.read(notify,4096);mask=struct.unpack_from('iIII',event)[1];assert mask&1\n"
        " stop_recorder();consumed=read_count()-before;assert 0<consumed<extent,consumed\n"
        " os.ftruncate(1,0)\n"
        " identity.with_suffix('.capture.json').write_text(json.dumps({'recorder':recorder,'start_tick':start,'stopped_state':'T','inotify_mask':mask,'read_bytes_before_truncation':consumed,'truncated_size':os.fstat(1).st_size}))\n"
        " os.close(notify);signal_recorder(signal.SIGCONT)\n"
        " while time.monotonic()<deadline:time.sleep(0.001)\n"
        "elif mode=='writer':\n"
        " identity.write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))\n"
        " deadline=time.monotonic()+budget\n"
        " guarded=False\n"
        " while time.monotonic()<deadline:\n"
        "  if not guarded and os.getppid()==guardian:\n"
        "   assert ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)==0\n"
        "   assert os.getppid()==guardian;guarded=True\n"
        "  os.ftruncate(1,extent);os.ftruncate(1,0)\n"
        "elif mode=='leader':\n"
        " subprocess.Popen([sys.executable,__file__,'writer',*sys.argv[2:]],start_new_session=True)\n"
        " deadline=time.monotonic()+budget\n"
        " while not identity.exists() and time.monotonic()<deadline:time.sleep(0.001)\n"
        " assert identity.exists()\n"
        " os.ftruncate(1,extent)\n"
        "else:\n"
        " from tools.facet_receipt import run_command\n"
        " status,counts,outcome,tool=run_command([sys.executable,__file__,'leader',*sys.argv[2:],str(os.getpid())],timeout=budget)\n"
        " result={'status':status,'counts':counts,'outcome':outcome,'stdout_length':len(tool['stdout']),'stderr':tool['stderr']}\n"
        " if protocol=='coordinated':result['capture']=json.loads(identity.with_suffix('.capture.json').read_text())\n"
        " print(json.dumps(result))\n"
    )
    extent = os.sysconf("SC_PAGE_SIZE") * (4096 if protocol == "coordinated" else 1024)
    controls: list[dict[str, object]] = []
    with _adopt_exited_child_descendants():
        for attempt in range(4):
            identity = tmp_path / f"writer-{attempt}.json"
            command = [
                sys.executable,
                str(script),
                "recorder",
                str(identity),
                str(os.getpid()),
                str(extent),
                str(budget),
                protocol,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            facts: dict[str, int] = {}
            reaped: tuple[int, int] | None = None
            try:
                stdout, stderr = process.communicate(timeout=2 * budget)
                assert process.returncode == 0, (process.returncode, stdout, stderr)
                result = json.loads(stdout)
                assert result["status"] == 0 and result["counts"]["passed"] == 0
                assert result["outcome"] == "skipped" and result["stderr"] == ""
                assert 0 <= result["stdout_length"] <= extent
                if protocol == "coordinated":
                    capture = result["capture"]
                    assert capture["recorder"] == process.pid
                    assert capture["stopped_state"] == "T" and capture["inotify_mask"] & 1
                    assert capture["truncated_size"] == 0
                    assert 0 < capture["read_bytes_before_truncation"] < extent
                    assert 0 < result["stdout_length"] < extent
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=budget)
                if identity.exists():
                    facts = json.loads(identity.read_text())
                    reaped = os.waitpid(facts["pid"], os.WNOHANG)
                    if reaped == (0, 0):
                        os.kill(facts["pid"], signal.SIGKILL)
                        reaped = os.waitpid(facts["pid"], 0)
            assert facts["pid"] == facts["pgid"]
            assert reaped is not None and reaped[0] == facts["pid"]
            assert os.waitstatus_to_exitcode(reaped[1]) == -signal.SIGKILL
            with pytest.raises(ProcessLookupError):
                os.kill(facts["pid"], 0)
            controls.append({"writer": facts, "reaped": reaped, "result": result})
    (tmp_path / "concurrent-output-control.json").write_text(
        json.dumps(
            {
                "observed": controls,
                "baseline_seconds": baseline,
                "budget_seconds": budget,
                "extent_bytes": extent,
                "protocol": protocol,
                "scientific_receipt_credit": False,
            },
            indent=2,
        )
        + "\n"
    )

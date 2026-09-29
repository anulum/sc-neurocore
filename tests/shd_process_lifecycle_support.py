# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native SHD lifecycle subprocess ownership

"""Run contained native-reader death tests without changing the pytest process attributes."""

import json
import os
import signal
import subprocess
import sys

_SHUTDOWN_PROBE = """
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contained native SHD lifecycle probe

import ctypes, json, os, select, signal, sys, time
from pathlib import Path

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
libc.prctl.restype = ctypes.c_int
assert libc.prctl(36, 1, 0, 0, 0) == 0, ctypes.get_errno()
read_end, write_end = os.pipe()
producer = os.fork()
if producer == 0:
    os.close(read_end)
    os.setpgid(0, 0)
    reader = os.fork()
    if reader == 0:
        os.close(write_end)
        command = json.loads(sys.argv[1]) + [str(os.getppid())]
        os.execv(command[0], command)
    os.write(write_end, str(reader).encode())
    os.close(write_end)
    signal.pause()
    os._exit(1)
os.close(write_end)
reader = None

def probe_timeout(signum, frame):
    '''Bound the contained probe while preserving its finally cleanup.'''
    raise TimeoutError('contained native reader probe exceeded 18 seconds')

signal.signal(signal.SIGALRM, probe_timeout)
signal.alarm(18)
try:
    assert select.select([read_end], [], [], 5)[0], 'reader pid was not delivered'
    reader = int(os.read(read_end, 64))
    deadline = time.monotonic() + 10
    while Path(f'/proc/{reader}/wchan').read_text().strip() != 'wait_for_partner':
        assert time.monotonic() < deadline, 'reader did not reach blocked HDF5 open'
        time.sleep(0.02)
    started = time.monotonic()
    if sys.argv[2] == "group":
        os.killpg(producer, signal.SIGTERM)
    else:
        os.kill(producer, signal.SIGKILL)
    assert os.waitpid(producer, 0)[0] == producer
    producer = None
    while True:
        pid, status = os.waitpid(reader, os.WNOHANG)
        if pid:
            reader = None
            expected = (-signal.SIGKILL, -signal.SIGTERM) if sys.argv[2] == 'group' else (-signal.SIGKILL,)
            assert os.waitstatus_to_exitcode(status) in expected, status
            print('owned reader reaped', flush=True)
            break
        assert time.monotonic() - started < 5, 'orphan reader survived parent death'
        time.sleep(0.02)
finally:
    signal.alarm(0)
    os.close(read_end)
    if producer is not None:
        os.kill(producer, signal.SIGKILL)
        os.waitpid(producer, 0)
    children = Path(f'/proc/self/task/{os.getpid()}/children')
    for child in children.read_text().split():
        pid = int(child)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
"""


def check_reader_shutdown(command: list[str], *, group: bool = False) -> None:
    """Exercise an actual blocked reader under a contained producer and subreaper.

    Parameters
    ----------
    command : list[str]
        Actual executable and recording arguments; expected parent is appended
        by its creating producer immediately before exec.
    group : bool
        Signal the producer's whole group instead of only its parent process.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _SHUTDOWN_PROBE,
            json.dumps(command),
            "group" if group else "parent",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        output, error = process.communicate(timeout=25)
        assert process.returncode == 0, error.decode()
        assert output == b"owned reader reaped\n"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()

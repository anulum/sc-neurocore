# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — supervisor pool shutdown from another Python thread

"""A running supervisor pool stops when another Python thread calls ``force_shutdown``.

``PySpikingControllerPool.start`` held the GIL for the whole control loop, so with no step
limit no other Python thread ran again, and ``force_shutdown`` could never be called. The loop
now runs without the GIL. The case runs in a child interpreter, so a regression shows as a
child that times out rather than a test run that hangs. A forced stop is reported as such:
``start`` raises a ``RuntimeError`` naming ``force_shutdown``, and the safety flag stays clear.

The pool's error place rises by design, so with the default snapshot period the verifier fails
the run's second or third snapshot within milliseconds, long before the forced stop. The child
therefore offers no snapshot (``snapshot_period=1 << 62``): the forced stop is then the only way
the loop can end, and its report does not depend on which thread wins a race.
"""

from __future__ import annotations

import subprocess
import sys

_CHILD = """
import threading, time
from sc_neurocore_engine.sc_neurocore_engine import PySpikingControllerPool
pool = PySpikingControllerPool(n_neurons=8, seed=7, snapshot_period=1 << 62)
outcome = {}
def run():
    try:
        outcome["steps"] = pool.start(max_steps=0)
    except RuntimeError as error:
        outcome["stopped"] = str(error)
worker = threading.Thread(target=run, daemon=True)
worker.start()
time.sleep(0.5)
deadline = time.monotonic() + 30.0
while worker.is_alive() and time.monotonic() < deadline:
    pool.force_shutdown()
    worker.join(timeout=0.1)
print("joined" if not worker.is_alive() else "still running", sorted(outcome))
print(outcome.get("stopped", ""))
print("tripped" if pool.is_safety_tripped() else "not tripped")
"""


def test_force_shutdown_from_another_thread_stops_an_unlimited_run() -> None:
    """With no step limit, the main thread still runs and its ``force_shutdown`` ends the loop."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "joined ['stopped']", result.stdout
    assert lines[1] == "Hardware execution stopped: force_shutdown was called.", result.stdout
    assert lines[2] == "not tripped", result.stdout

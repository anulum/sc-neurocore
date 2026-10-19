# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — checking whether a network can go native leaves the caller's generators alone

"""The native-dispatch check neither moves nor hides the process-wide generators.

The state probe stepped a default neuron 44 times without restoring the global
generators. ``StochasticIFNeuron`` draws from ``np.random``, so a run with
``backend="auto"`` (or a refused ``backend="rust"``) advanced the caller's
stream by 44 draws and changed every later simulation (RIGOR
NEURO-RUST-DISPATCH-RNG-01: final v -70.578 instead of -69.477 at seed 1729).
Construction and stepping in the probe now run inside a generator transaction,
and a model that draws from a process-wide generator is not dispatched natively
(a native run would not advance that generator either).

Each case runs in a fresh interpreter, so the cached probe is cold, as in the
finding; a second network in the same interpreter covers the warm check. The
auto run must equal the direct Python run step for step and leave the
generator at the same state; a refused forced-Rust run must leave it untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]

_SCRIPT = """
import json, sys
import numpy as np
from sc_neurocore.network import Network, Population
from sc_neurocore.network.monitor import StateMonitor

seed, backend = int(sys.argv[1]), sys.argv[2]

def run(backend):
    population = Population("StochasticIFNeuron", 2)
    monitor = StateMonitor(population)
    network = Network(population, monitor)
    try:
        network.run(0.005, dt=0.001, backend=backend)
    except Exception as error:
        return {"refused": type(error).__name__}
    return {"v": [float(n.v) for n in population.neurons]}

np.random.seed(seed)
first = run(backend)
after_first = np.random.random(4).tolist()
second = run(backend)  # warm: the probe is cached now
after_second = np.random.random(4).tolist()
print(json.dumps({"first": first, "second": second, "draws": after_first + after_second}))
"""


def _run(seed: int, backend: str) -> dict[str, object]:
    """Run the script in a fresh interpreter and return its JSON record."""
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(seed), backend],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO,
        env={"PYTHONPATH": str(_REPO / "src"), "PATH": ""},
        check=True,
    )
    record: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return record


@pytest.mark.parametrize("seed", [1729, 42, 2026])
def test_auto_equals_python_and_leaves_the_stream(seed: int) -> None:
    """Cold and warm auto runs give the Python trajectories and the Python next draws."""
    python = _run(seed, "python")
    auto = _run(seed, "auto")
    first = python["first"]
    assert isinstance(first, dict) and "v" in first
    assert auto == python


@pytest.mark.parametrize("seed", [1729, 2026])
def test_a_refused_native_run_leaves_the_stream(seed: int) -> None:
    """Forced Rust refuses (StateMonitor, global draws) and the next draws are untouched."""
    forced = _run(seed, "rust")
    first, second = forced["first"], forced["second"]
    assert isinstance(first, dict) and "refused" in first
    assert isinstance(second, dict) and "refused" in second
    np.random.seed(seed)
    assert forced["draws"] == np.random.random(8).tolist()

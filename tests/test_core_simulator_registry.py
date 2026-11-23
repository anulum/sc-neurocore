# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simulator registry behaviour through real package metadata

"""Exercise :mod:`sc_neurocore.core.backends` through real installed metadata.

Each refusal case runs in a child interpreter whose path starts with distribution metadata
written to disk for that case. ``importlib.metadata`` takes the first distribution of a name it
finds, so a case's own ``sc-neurocore`` metadata stands in for the installed one. The
no-metadata case masks the installed ``sc-neurocore`` metadata out of every site directory, as
a checkout that was never installed has none. Nothing is mocked.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sc_neurocore.core import backends
from sc_neurocore.neurons.models.alpha import AlphaNeuron

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

CHILD = r"""
import json, sys
sys.path[:0] = json.loads(sys.argv[1])
from sc_neurocore.core import backends
model_id = sys.argv[2]
rewrite = json.loads(sys.argv[3])
outcome = []
for step in ("first", "cached", "cleared"):
    if step == "cleared":
        if rewrite:
            open(rewrite[0], "w", encoding="utf-8").write(rewrite[1])
        backends.clear_simulator_cache()
    try:
        simulator = backends.resolve_simulator(model_id)
    except backends.SimulatorRegistryError as error:
        outcome.append({"refused": type(error).__name__, "message": str(error),
                        "cause": type(error.__cause__).__name__ if error.__cause__ else None})
        break
    outcome.append({"resolved": simulator.__module__ + ":" + simulator.__qualname__,
                    "result": simulator()})
print(json.dumps(outcome))
"""


def _distribution(root: Path, name: str, entries: dict[str, str]) -> Path:
    """Write a distribution's metadata declaring ``entries`` in the simulators group."""
    info = root / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    lines = ["[sc_neurocore.simulators]", *(f"{key} = {value}" for key, value in entries.items())]
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info / "entry_points.txt"


def _module(root: Path, name: str, body: str) -> None:
    """Write an importable module of the case's own simulators."""
    (root / f"{name}.py").write_text(body, encoding="utf-8")


def _run(
    paths: list[Path], model_id: str, rewrite: tuple[Path, str] | None = None
) -> list[dict[str, object]]:
    """Resolve ``model_id`` in a child interpreter whose path starts with ``paths``."""
    argv = [
        sys.executable,
        "-c",
        CHILD,
        json.dumps([str(path) for path in [*paths, SRC]]),
        model_id,
        json.dumps([str(rewrite[0]), rewrite[1]] if rewrite else None),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120)
    outcome: list[dict[str, object]] = json.loads(completed.stdout.strip().splitlines()[-1])
    return outcome


SIMULATORS = (
    "def first():\n    return 'first'\n\n\ndef second():\n    return 'second'\n\n\nVALUE = 3\n"
)


def test_an_allowed_registration_resolves_and_is_cached_until_cleared(tmp_path: Path) -> None:
    """The declared callable is returned, kept across calls and read again after a clear."""
    _module(tmp_path, "case_simulators", SIMULATORS)
    entry_points = _distribution(tmp_path, "sc-neurocore", {"m": "case_simulators:first"})
    outcome = _run(
        [tmp_path],
        "m",
        rewrite=(entry_points, "[sc_neurocore.simulators]\nm = case_simulators:second\n"),
    )
    assert outcome == [
        {"resolved": "case_simulators:first", "result": "first"},
        {"resolved": "case_simulators:first", "result": "first"},
        {"resolved": "case_simulators:second", "result": "second"},
    ]


def test_the_metadata_is_scanned_once_per_process_until_cleared(tmp_path: Path) -> None:
    """A model id added to the metadata after the first resolution is seen only after a clear."""
    _module(tmp_path, "case_simulators", SIMULATORS)
    entry_points = _distribution(tmp_path, "sc-neurocore", {"m": "case_simulators:first"})
    script = (
        "import json, sys\n"
        "sys.path[:0] = json.loads(sys.argv[1])\n"
        "from sc_neurocore.core import backends\n"
        "backends.resolve_simulator('m')\n"
        "open(sys.argv[2], 'a', encoding='utf-8').write('n = case_simulators:second\\n')\n"
        "try:\n"
        "    backends.resolve_simulator('n')\n"
        "except backends.NoSimulatorRegistered:\n"
        "    print('not seen before the clear')\n"
        "backends.clear_simulator_cache()\n"
        "print('after the clear:', backends.resolve_simulator('n')())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps([str(tmp_path), str(SRC)]), str(entry_points)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert completed.stdout.splitlines()[-2:] == [
        "not seen before the clear",
        "after the clear: second",
    ]


def test_two_registrations_of_one_id_are_refused_naming_both(tmp_path: Path) -> None:
    """A second distribution declaring the same id is a refusal, never an order-based pick."""
    _module(tmp_path, "case_simulators", SIMULATORS)
    _distribution(tmp_path, "sc-neurocore", {"m": "case_simulators:first"})
    _distribution(tmp_path, "other-plugin", {"m": "case_simulators:second"})
    (refusal,) = _run([tmp_path], "m")
    assert refusal["refused"] == "DuplicateSimulatorRegistration"
    message = str(refusal["message"])
    assert "case_simulators:first (from sc-neurocore)" in message
    assert "case_simulators:second (from other-plugin)" in message


def test_a_registration_outside_the_allowlist_does_not_count(tmp_path: Path) -> None:
    """Another distribution cannot supply a model's simulator; the ignored one is named."""
    _module(tmp_path, "case_simulators", SIMULATORS)
    _distribution(tmp_path, "sc-neurocore", {})
    _distribution(tmp_path, "intruder", {"m": "case_simulators:second"})
    (refusal,) = _run([tmp_path], "m")
    assert refusal["refused"] == "NoSimulatorRegistered"
    assert "ignored m = case_simulators:second (from intruder)" in str(refusal["message"])


def test_an_unregistered_id_is_refused(tmp_path: Path) -> None:
    """An id that no distribution declares is a typed refusal."""
    _distribution(tmp_path, "sc-neurocore", {})
    (refusal,) = _run([tmp_path], "unknown-model")
    assert refusal["refused"] == "NoSimulatorRegistered"
    assert "'unknown-model'" in str(refusal["message"])


def test_a_registration_that_fails_to_import_is_refused_with_its_cause(tmp_path: Path) -> None:
    """A load error names the entry point and keeps the original error as the cause."""
    _module(tmp_path, "broken_simulators", "raise ImportError('native library missing')\n")
    _distribution(tmp_path, "sc-neurocore", {"m": "broken_simulators:run"})
    (refusal,) = _run([tmp_path], "m")
    assert refusal["refused"] == "SimulatorLoadRefused"
    assert refusal["cause"] == "ImportError"
    assert "m = broken_simulators:run (from sc-neurocore)" in str(refusal["message"])
    assert "native library missing" in str(refusal["message"])


def test_a_missing_attribute_and_a_non_callable_are_refused(tmp_path: Path) -> None:
    """Both a name the module lacks and a value that is not callable are refusals."""
    _module(tmp_path, "case_simulators", SIMULATORS)
    _distribution(
        tmp_path, "sc-neurocore", {"a": "case_simulators:absent", "v": "case_simulators:VALUE"}
    )
    (missing,) = _run([tmp_path], "a")
    assert missing["refused"] == "SimulatorLoadRefused"
    assert missing["cause"] == "AttributeError"
    (value,) = _run([tmp_path], "v")
    assert value["refused"] == "SimulatorLoadRefused"
    assert "loaded a int, not a callable" in str(value["message"])


def test_a_checkout_without_installed_metadata_is_refused(tmp_path: Path) -> None:
    """Hide every sc-neurocore metadata directory, as a checkout that was never installed has none.

    The child sees the package sources through a directory that holds only the package, every
    site directory without the sc-neurocore metadata, and the standard library. A metadata
    directory beside the sources, such as a leftover ``src/*.egg-info``, stays out of view.
    """
    sources = tmp_path / "src"
    sources.mkdir()
    (sources / "sc_neurocore").symlink_to(SRC / "sc_neurocore")
    masked: list[Path] = [sources]
    for index, entry in enumerate(p for p in sys.path if p.endswith("site-packages")):
        site = Path(entry)
        if not site.is_dir():
            continue
        mask = tmp_path / f"site{index}"
        mask.mkdir()
        for child in site.iterdir():
            hidden = child.name.startswith(("sc_neurocore-", "__editable__.sc_neurocore-"))
            if not hidden:
                (mask / child.name).symlink_to(child)
        masked.append(mask)
    standard = [
        entry
        for entry in sys.path
        if entry.startswith((sys.base_prefix, sys.base_exec_prefix))
        and not entry.endswith("site-packages")
    ]
    script = (
        "import json, sys\n"
        "sys.path[:] = json.loads(sys.argv[1])\n"
        "from sc_neurocore.core import backends\n"
        "try:\n"
        "    backends.resolve_simulator('alpha')\n"
        "except backends.NoDistributionMetadata as error:\n"
        "    print('refused:', error)\n"
    )
    paths = json.dumps([*(str(mask) for mask in masked), *standard])
    completed = subprocess.run(
        [sys.executable, "-c", script, paths],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert "refused: no backend registered for 'alpha': no installed distribution metadata" in (
        completed.stdout
    )


def test_the_installed_alpha_registration_is_the_engine_simulator() -> None:
    """In this installed package, the alpha id resolves to the engine's batch simulator."""
    from sc_neurocore.accel.alpha import simulate_alpha

    backends.clear_simulator_cache()
    assert backends.resolve_simulator("alpha") is simulate_alpha
    assert backends.resolve_simulator("alpha") is simulate_alpha


def test_the_alpha_model_runs_through_the_registry_unchanged() -> None:
    """``AlphaNeuron.simulate`` gives the result of calling the engine simulator directly."""
    from sc_neurocore.accel.alpha import simulate_alpha

    current = np.linspace(0.0, 3.0, 64)
    neuron = AlphaNeuron()
    expected = simulate_alpha(
        neuron.v,
        neuron.a_exc,
        neuron.i_exc,
        neuron.a_inh,
        neuron.i_inh,
        neuron.v_rest,
        neuron.v_threshold,
        neuron.tau_v,
        neuron.tau_exc,
        neuron.tau_inh,
        neuron.dt,
        current,
        0.0,
        backend="python",
    )
    result = neuron.simulate(current, backend="python")
    assert np.array_equal(result["spikes"], expected["spikes"])
    assert neuron.v == float(expected["v_final"])


def test_the_alpha_model_does_not_import_its_executor() -> None:
    """The model module names the registry, and no ``sc_neurocore.accel`` module."""
    source = (SRC / "sc_neurocore" / "neurons" / "models" / "alpha.py").read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "sc_neurocore.core.backends" in imported
    assert not any(name.startswith("sc_neurocore.accel") for name in imported)

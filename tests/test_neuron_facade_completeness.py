# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — neuron facade completeness and Rust dispatch allow-list

"""The ``sc_neurocore.neurons`` facade exports every model and dispatches to Rust only by list.

Every class in ``sc_neurocore.neurons.models.__all__`` must be importable from the facade. A
name resolves to its Rust class only when it is in ``_RUST_DISPATCH_NAMES``, and each listed
name must keep two properties: the Rust constructor accepts every Python constructor field at its
default, and 500 steps of the parity harness's stimulus give the same outputs as the Python class.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import tests.test_rust_python_neuron_parity as harness
from sc_neurocore import neurons as facade
from sc_neurocore.neurons import models as py_models
from tests.engine_requirement import require_engine

_ROOT = Path(__file__).resolve().parents[1]
_STEPS = 500


def _defaults(cls: type) -> dict[str, object]:
    """Return the public constructor fields of a Python model with their default values."""
    if dataclasses.is_dataclass(cls):
        values: dict[str, object] = {}
        for field in dataclasses.fields(cls):
            if not field.init or field.name.startswith("_"):
                continue
            if field.default is not dataclasses.MISSING:
                values[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                values[field.name] = field.default_factory()
        return values
    signature = inspect.signature(cls)
    return {
        key: parameter.default
        for key, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty and not key.startswith("_")
    }


def _outputs(model: object, name: str, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Run the parity harness stimulus for ``_STEPS`` steps and record every raw step output."""
    seen: list[object] = []

    def record(value: object) -> int:
        """Keep the raw output and report no spike to the harness."""
        seen.append(value)
        return 0

    monkeypatch.setattr(harness, "_spike_count", record)
    harness._run_steps(cast(harness.StepModel, model), name, _STEPS)
    return seen


def _as_list(value: object) -> list[object]:
    """Return a sequence or array output as a list."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return list(cast(list[object], tolist()))
    return list(cast(list[object], value))


def _same(left: object, right: object) -> bool:
    """Compare two outputs: exactly for integers, to rel 1e-9 for floats, elementwise for sequences."""
    if isinstance(left, (list, tuple)) or hasattr(left, "tolist"):
        a, b = _as_list(left), _as_list(right)
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        fa, fb = float(cast(float, left)), float(cast(float, right))
    except (TypeError, ValueError):
        return left == right
    if math.isnan(fa) and math.isnan(fb):
        return True
    return fa == fb or abs(fa - fb) <= 1e-9 * max(abs(fa), abs(fb))


def test_facade_exports_every_model() -> None:
    """The facade's model names are exactly the model registry's, apart from the core names."""
    assert set(py_models.__all__) - facade._CORE_NAMES == facade._MODEL_NAMES
    assert set(py_models.__all__) <= set(facade.__all__)


def test_dispatch_list_is_one_frozenset_of_facade_names() -> None:
    """The allow-list is a frozenset drawn from the facade's model names."""
    assert isinstance(facade._RUST_DISPATCH_NAMES, frozenset)
    assert facade._RUST_DISPATCH_NAMES <= facade._MODEL_NAMES


@pytest.mark.parametrize("name", sorted(facade._RUST_DISPATCH_NAMES))
def test_listed_name_keeps_constructor_and_trajectory_parity(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed Rust class takes every Python field and reproduces 500 steps of Python output."""
    require_engine()
    engine = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    rust_cls = cast(type, getattr(engine, name))
    py_cls = cast(type, getattr(py_models, name))
    rust_cls(**_defaults(py_cls))
    python_outputs = _outputs(py_cls(), name, monkeypatch)
    rust_outputs = _outputs(rust_cls(), name, monkeypatch)
    assert len(python_outputs) == len(rust_outputs) == _STEPS
    for step, (py_value, rs_value) in enumerate(zip(python_outputs, rust_outputs)):
        assert _same(py_value, rs_value), (name, step, py_value, rs_value)


def _resolved_modules(env_extra: dict[str, str]) -> dict[str, str]:
    """Resolve every facade model name in a fresh interpreter and return each class's module."""
    code = (
        "import json\nfrom sc_neurocore import neurons\n"
        "print(json.dumps({n: getattr(neurons, n).__module__ for n in sorted(neurons._MODEL_NAMES)}))\n"
    )
    env = {**os.environ, **env_extra}
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), str(_ROOT / "src"), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    ).stdout
    return cast(dict[str, str], json.loads(out.strip().splitlines()[-1]))


def test_only_listed_names_dispatch_to_rust() -> None:
    """With the engine installed and dispatch enabled, exactly the listed names resolve to Rust."""
    require_engine()
    modules = _resolved_modules({"SC_NEUROCORE_NO_RUST": ""})
    rust = {name for name, module in modules.items() if module.startswith("sc_neurocore_engine")}
    assert rust == set(facade._RUST_DISPATCH_NAMES)


def test_no_rust_flag_disables_every_dispatch() -> None:
    """With SC_NEUROCORE_NO_RUST=1 every facade name resolves to its Python class."""
    modules = _resolved_modules({"SC_NEUROCORE_NO_RUST": "1"})
    assert not [
        name for name, module in modules.items() if module.startswith("sc_neurocore_engine")
    ]

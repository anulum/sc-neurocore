# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Li-Rinzel astrocyte Rust and Python parity

"""The Rust engine's ``AstrocyteModel`` against the Python Li-Rinzel reference.

The Python model has three ODEs (cytosolic calcium, the IP3-receptor gate and IP3), an ER
leak at rate ``leak = 0.01`` per second, and an input that drives IP3 production. It refuses
a negative or non-finite input and a calcium candidate outside the total calcium pool. The
Rust class must give the same trajectory and the same refusals.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from typing import Protocol, cast

import pytest

from sc_neurocore.neurons.models.astrocyte import AstrocyteModel
from tests.julia_requirement import require_julia_executable

_HAS_ENGINE = importlib.util.find_spec("sc_neurocore_engine") is not None
_JULIA_KERNEL = (
    Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/neurons/astrocyte.jl"
)
_STATE = ("ca", "h", "ip3")


class RustAstrocyte(Protocol):
    """The public surface of the Rust engine's ``AstrocyteModel``."""

    def step(self, current: float) -> float:
        """Advance one Euler step and return the cytosolic calcium."""

    def reset(self) -> None:
        """Restore the default state."""

    def get_state(self) -> dict[str, float]:
        """Return ``ca``, ``h`` and ``ip3``."""


def _rust() -> RustAstrocyte:
    """Return a new Rust ``AstrocyteModel``, or skip when the engine is not built."""
    if not _HAS_ENGINE:
        pytest.skip("Rust engine not built")
    engine = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    return cast(RustAstrocyte, engine.AstrocyteModel())


def _python_state(model: AstrocyteModel) -> tuple[float, float, float]:
    """Return the Python model's (ca, h, ip3)."""
    return float(model.ca), float(model.h), float(model.ip3)


def _rust_state(model: RustAstrocyte) -> tuple[float, float, float]:
    """Return the Rust model's (ca, h, ip3) from ``get_state``."""
    state = model.get_state()
    return float(state["ca"]), float(state["h"]), float(state["ip3"])


@pytest.mark.parametrize("current", [0.0, 0.1, 2.5])
def test_rust_trajectory_matches_python(current: float) -> None:
    """Over 500 steps the Rust calcium, gate and IP3 follow Python's to 1e-12."""
    python = AstrocyteModel()
    rust = _rust()
    for step in range(500):
        returned_py = python.step(current)
        returned_rs = rust.step(current)
        assert returned_rs == pytest.approx(returned_py, rel=1e-12, abs=1e-15), step
        for name, py_value, rs_value in zip(_STATE, _python_state(python), _rust_state(rust)):
            assert rs_value == pytest.approx(py_value, rel=1e-12, abs=1e-15), (step, name)


def test_input_drives_ip3_not_calcium() -> None:
    """After 10 steps at 2.5 IP3 has risen from 0.5 to about 0.74 in both lanes."""
    python = AstrocyteModel()
    rust = _rust()
    for _ in range(10):
        python.step(2.5)
        rust.step(2.5)
    ca_py, _, ip3_py = _python_state(python)
    ca_rs, _, ip3_rs = _rust_state(rust)
    assert ip3_py == pytest.approx(0.741475, abs=5e-7)
    assert ca_py == pytest.approx(0.075363, abs=5e-7)
    assert (ca_rs, ip3_rs) == pytest.approx((ca_py, ip3_py), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("current", [-0.1, math.nan, math.inf])
def test_invalid_input_is_refused_and_leaves_the_state(current: float) -> None:
    """A negative or non-finite input raises ValueError in both lanes, and nothing changes."""
    python = AstrocyteModel()
    rust = _rust()
    before_py = _python_state(python)
    before_rs = _rust_state(rust)
    with pytest.raises(ValueError, match="current"):
        python.step(current)
    with pytest.raises(ValueError, match="current"):
        rust.step(current)
    assert _python_state(python) == before_py
    assert _rust_state(rust) == before_rs


def test_reset_restores_the_class_defaults() -> None:
    """After a run, reset returns both lanes to ca 0.05, h 0.8 and IP3 0.5."""
    python = AstrocyteModel()
    rust = _rust()
    for _ in range(50):
        python.step(1.0)
        rust.step(1.0)
    python.reset()
    rust.reset()
    assert _python_state(python) == (0.05, 0.8, 0.5)
    assert _rust_state(rust) == (0.05, 0.8, 0.5)


def test_julia_lane_matches_python_bit_for_bit() -> None:
    """Over 200 steps at 2.5 the Julia lane's state equals Python's exactly, and it refuses -0.1."""
    julia = require_julia_executable()
    expression = (
        f'include(raw"{_JULIA_KERNEL}"); using .AstrocyteAccel; s = AstrocyteModelState(); '
        "for _ in 1:200; step!(s, 2.5); end; "
        "refused = step!(AstrocyteModelState(), -0.1) === nothing; "
        'print("[", s.ca, ",", s.h, ",", s.ip3, ",", refused, "]")'
    )
    output = subprocess.run(
        [julia, "--startup-file=no", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    values = json.loads(output)
    python = AstrocyteModel()
    for _ in range(200):
        python.step(2.5)
    assert values[:3] == list(_python_state(python))
    assert values[3] is True

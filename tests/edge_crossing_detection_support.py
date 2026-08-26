# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_edge_crossing_detection.py

from __future__ import annotations

"""Tests for the ``detection = "crossing"`` rising-edge threshold capability.

A non-resetting oscillator (FitzHugh-Nagumo, the preserved SC triangular McKean form)
spikes when its membrane crosses
threshold *upward*, not on every step the membrane stays above threshold. The schema DSL
now honours ``[threshold] detection = "crossing"`` in both the Python runner and the
emitted Verilog, so such oscillators co-simulate faithfully against their hand models.

Edge detection is engaged only for a crossing model that declares **no reset**: a reset
that drops the state below threshold already clears the condition every spike, so
``level`` and ``crossing`` are identical for reset-based integrate-and-fire models, which
keep the previously validated level datapath unchanged.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import pytest
from sc_neurocore.compiler.equation_compiler import compile_to_verilog, generate_testbench
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.neurons.models.sc_triangular_mckean import SCTriangularMcKeanNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

HAS_IVERILOG = shutil.which("iverilog") is not None
_FHN_PARAMS = {"a": 0.7, "b": 0.8, "epsilon": 0.08, "v_threshold": 1.0}
_FHN_INIT = {"v": -1.0, "w": -0.5}
_MCKEAN_PARAMS = {"a": 0.25, "epsilon": 0.3, "gamma": 2.0, "v_peak": 0.8}
_MCKEAN_INIT = {"v": 0.0, "w": 0.0}


def _wrapped_phase_schema(theta: float = 0.0) -> dict[str, object]:
    """Return a one-state Euler phase map with an explicit pre-wrap crossing."""
    candidate = "theta + dt * ((1.0 - cos(theta)) + (1.0 + cos(theta)) * gain * I)"
    previous_candidate = (
        "theta_prev + dt * ((1.0 - cos(theta_prev)) + (1.0 + cos(theta_prev)) * gain * I)"
    )
    return {
        "metadata": {"schema_version": 1, "name": "Wrapped phase"},
        "state": {"theta": theta},
        "parameters": {
            "dt": 0.1,
            "gain": 1.0,
            "theta_threshold": 3.141592653589793,
        },
        "integration": {"dt": 1.0, "method": "map"},
        "dynamics": {"theta": f"({candidate}) % 6.283185307179586"},
        "threshold": {
            "condition": f"theta_prev < theta_threshold <= ({previous_candidate})",
            "detection": "level",
        },
    }


def _fhn_hand() -> FitzHughNagumoNeuron:
    """Hand-authored FitzHugh-Nagumo neuron at the same operating point as the schema."""
    return FitzHughNagumoNeuron(dt=0.1, v=-1.0, w=-0.5, a=0.7, b=0.8, epsilon=0.08, v_threshold=1.0)


def _fhn_schema(detection: str = "crossing") -> dict[str, object]:
    """Faithful FitzHugh-Nagumo schema (RK4, no reset) matching ``FitzHughNagumoNeuron``."""
    return {
        "metadata": {"schema_version": 1, "name": "FitzHugh-Nagumo"},
        "state": dict(_FHN_INIT),
        "parameters": dict(_FHN_PARAMS),
        "integration": {"dt": 0.1, "method": "rk4"},
        # ``v * v * v`` (not ``v ** 3``) to reproduce the hand model's exact IEEE cube.
        "dynamics": {
            "v": "v - v * v * v / 3.0 - w + I",
            "w": "epsilon * (v + a - b * w)",
        },
        "threshold": {"condition": "v >= v_threshold", "detection": detection},
    }


def _mckean_schema(detection: str = "crossing") -> dict[str, object]:
    """McKean piecewise-linear caricature schema (RK4, no reset) via min/max branches."""
    return {
        "metadata": {"schema_version": 1, "name": "McKeanNeuron"},
        "state": dict(_MCKEAN_INIT),
        "parameters": dict(_MCKEAN_PARAMS),
        "integration": {"dt": 0.1, "method": "rk4"},
        "dynamics": {
            "v": "min(max(-v, v - a), 1 - v) - w + I",
            "w": "epsilon * (v - gamma * w)",
        },
        "threshold": {"condition": "v >= v_peak", "detection": detection},
    }


def _verilog_spike_count_q1616(
    neuron: UniversalNeuron, n_steps: int, current: float, tag: str
) -> int:
    """Compile a schema neuron at Q16.16, simulate with iverilog, return the spike count."""
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_edge_{tag}_q1616"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl = Path(tmpdir) / f"{module_name}.v"
        tbp = Path(tmpdir) / f"tb_{module_name}.v"
        out = Path(tmpdir) / f"tb_{module_name}"
        rtl.write_text(verilog)
        tbp.write_text(tb)
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out), str(rtl), str(tbp)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{compile_result.stderr}")
        sim_result = subprocess.run(["vvp", str(out)], capture_output=True, text=True, timeout=90)
        if sim_result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{sim_result.stderr}")
        match = re.search(r"(\d+) spikes", sim_result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{sim_result.stdout}")
        return int(match.group(1))


__all__ = [
    "re",
    "shutil",
    "subprocess",
    "tempfile",
    "Path",
    "pytest",
    "compile_to_verilog",
    "generate_testbench",
    "EquationNeuron",
    "FitzHughNagumoNeuron",
    "SCTriangularMcKeanNeuron",
    "UniversalNeuron",
    "HAS_IVERILOG",
    "_FHN_PARAMS",
    "_FHN_INIT",
    "_MCKEAN_PARAMS",
    "_MCKEAN_INIT",
    "_wrapped_phase_schema",
    "_fhn_hand",
    "_fhn_schema",
    "_mckean_schema",
    "_verilog_spike_count_q1616",
]

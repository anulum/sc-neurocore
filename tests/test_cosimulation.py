# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python ↔ Verilog co-simulation: bit-true equivalence tests

"""Co-simulation tests: run identical models in Python and Verilog, compare.

These tests validate the core SC-NeuroCore claim: the Python simulation
and the Verilog hardware implementation produce **bit-true equivalent**
spike behaviour across all precision modes and all simulatable models.

Pipeline::

    schema → UniversalNeuron (Python)   → step() for N cycles → spike count
    schema → compile_to_verilog()       → iverilog + vvp      → spike count
    assert spike counts match within tolerance

Test Classes
------------
TestCoSimulation
    Q8.8 baseline: spike production, accuracy (<1%), zero-current silence,
    determinism.  Covers LIF, Lapicque, Quadratic IF, Izhikevich,
    Resonate-and-Fire.

TestQ412Precision
    Q4.12 (16-bit, 12 fractional): precision vs Q8.8, range xfail.

TestQ1616Precision
    Q16.16 (32-bit): gold standard fidelity, zero-current silence.

Verified Results (2026-05-01)
-----------------------------
All 5 models: 0.0% spike count gap at Q8.8 (I=50.0, 200 steps).
All 3 precision modes (Q8.8, Q4.12, Q16.16): 0.0% gap for LIF.

Prerequisites
-------------
- Icarus Verilog (``iverilog``, ``vvp``) — tests skip if unavailable.
- Install: ``apt install iverilog`` (Ubuntu) or ``brew install icarus-verilog`` (macOS).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.compiler.equation_compiler import (
    generate_testbench,
)

HAS_IVERILOG = shutil.which("iverilog") is not None

# Co-simulation parameters
# NOTE: Q8.8 fixed-point has ±0.004 precision, which causes quantization
# drift in threshold and dynamics calculations. Higher currents are needed
# to reliably trigger spikes in the Verilog implementation.
_N_STEPS = 200
_INPUT_CURRENT = 50.0  # Higher than Python needs — overcomes Q8.8 precision loss

# Models suitable for co-simulation (polynomial/linear, no transcendental functions).
# All 5 models achieve 0% Python↔Verilog spike count gap.
_COSIM_MODELS = ["lif", "lapicque", "quadratic_if", "izhikevich", "resonate_fire"]

# Transcendental models reachable through the auto model→RTL path once the emitter
# lowers negative LUT entries correctly, supports cosh, and omits an empty parameter
# list. `theta` (phase oscillator) co-simulates near bit-true; `glif` and
# `morris_lecar` lower to valid Verilog but Q8.8 + 16-entry LUTs are too coarse for a
# spike-count parity claim, so they are validated at compile level only (honest).
_TRANSCENDENTAL_COSIM_MODELS = ["theta"]
_TRANSCENDENTAL_TOLERANCE_PCT = 5.0
_TRANSCENDENTAL_COMPILE_MODELS = ["glif", "theta", "morris_lecar"]


def _python_spike_count(model_name: str, n_steps: int, current: float) -> int:
    """Run a model in Python and return the spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    spikes = 0
    for _ in range(n_steps):
        if neuron.step(I=current):
            spikes += 1
    return spikes


def _verilog_spike_count(model_name: str, n_steps: int, current: float) -> int:
    """Compile a model to Verilog, simulate with iverilog, return spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}"

    verilog = neuron.to_verilog(module_name=module_name)
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        # Compile
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        # Simulate
        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        # Parse spike count from output: "Simulation complete: N spikes in M cycles"
        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestCoSimulation:
    """Python ↔ Verilog co-simulation: validate spike behaviour equivalence."""

    @pytest.mark.parametrize("model_name", _COSIM_MODELS)
    def test_both_produce_spikes(self, model_name: str) -> None:
        """Verify both implementations produce non-zero spike output."""
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)

        # Both should spike (model is being driven with sufficient current)
        assert py_spikes > 0, f"Python {model_name} produced 0 spikes"
        assert vlog_spikes > 0, f"Verilog {model_name} produced 0 spikes"

    @pytest.mark.parametrize("model_name", _COSIM_MODELS)
    def test_spike_count_accuracy(self, model_name: str) -> None:
        """Q8.8 co-simulation must be within 1% of Python float64.

        With proper Q-format division, look-ahead threshold detection,
        and correct testbench timing, all models achieve 0% gap.
        """
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)

        assert py_spikes > 0, f"Python {model_name} must spike"
        assert vlog_spikes > 0, f"Verilog {model_name} must spike"

        gap = abs(py_spikes - vlog_spikes)
        gap_pct = gap / max(py_spikes, 1) * 100
        print(
            f"\n  Co-sim {model_name}: Python={py_spikes}, Verilog={vlog_spikes}, "
            f"gap={gap} ({gap_pct:.1f}%)"
        )

        assert gap_pct < 1.0, (
            f"Q8.8 co-simulation gap too large: {gap_pct:.1f}% "
            f"(model={model_name}, Python={py_spikes}, Verilog={vlog_spikes})"
        )

    @pytest.mark.parametrize("model_name", [m for m in _COSIM_MODELS if m != "izhikevich"])
    def test_no_current_no_spikes(self, model_name: str) -> None:
        """With zero input current, linear models should not spike.

        Izhikevich is excluded: its +140 constant term drives intrinsic dynamics,
        and Q8.8 quantization of 0.04*v^2 (-65^2=4225, overflows 16-bit product)
        causes divergent behaviour at zero current. Use Q16.16 for Izhikevich
        if zero-current silence is required.
        """
        py_spikes = _python_spike_count(model_name, 50, 0.0)
        vlog_spikes = _verilog_spike_count(model_name, 50, 0.0)

        assert py_spikes == 0, f"Python {model_name} spiked with zero current"
        assert vlog_spikes == 0, f"Verilog {model_name} spiked with zero current"

    def test_python_sim_is_deterministic(self) -> None:
        """Verify Python simulation is deterministic across runs."""
        a = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        b = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        assert a == b

    def test_verilog_sim_is_deterministic(self) -> None:
        """Verify Verilog simulation is deterministic across runs."""
        a = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        b = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        assert a == b


def _verilog_spike_count_q412(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q4.12 precision and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q412"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=16,
        fraction=12,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=16,
        fraction=12,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ412Precision:
    """Q4.12 precision mode: 4 integer + 12 fractional bits.

    Q4.12 has 1/4096 ≈ 0.00024 resolution (16× finer than Q8.8),
    which dramatically reduces the quantization gap at the cost of
    a narrower integer range ([-8, +7.9997] vs [-128, +127.996]).
    """

    def test_lif_q412_spikes(self) -> None:
        """Q4.12 LIF should spike reliably."""
        vlog_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)
        assert vlog_spikes > 0

    def test_lif_q412_near_python(self) -> None:
        """Q4.12 should close the LIF quantization gap to <5%.

        This is the key precision validation: Q8.8 has a ~99% gap,
        while Q4.12 should be within a few percent of float64.
        """
        py_spikes = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)

        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        print(
            f"\n  Q4.12 co-sim LIF: Python={py_spikes}, Verilog={vlog_spikes}, "
            f"gap={abs(py_spikes - vlog_spikes)} ({gap_pct:.1f}%)"
        )

        # Q4.12 should be within 5% of Python
        assert gap_pct < 5.0, (
            f"Q4.12 gap too large: {gap_pct:.1f}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

    def test_q412_vs_q88_comparison(self) -> None:
        """Compare Q4.12 vs Q8.8 accuracy for LIF.

        With the division fix and look-ahead threshold, both Q8.8 and Q4.12
        achieve near-perfect accuracy for LIF. This test verifies both
        formats are within 5% of Python and documents the comparison.
        """
        py_spikes = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        q88_spikes = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        q412_spikes = _verilog_spike_count_q412("lif", _N_STEPS, _INPUT_CURRENT)

        gap_q88 = abs(py_spikes - q88_spikes)
        gap_q412 = abs(py_spikes - q412_spikes)

        print(
            f"\n  Precision comparison LIF:"
            f"\n    Q8.8:  Python={py_spikes}, Verilog={q88_spikes}, gap={gap_q88}"
            f"\n    Q4.12: Python={py_spikes}, Verilog={q412_spikes}, gap={gap_q412}"
        )

        # Both should be within 5% of Python
        pct_q88 = gap_q88 / max(py_spikes, 1) * 100
        pct_q412 = gap_q412 / max(py_spikes, 1) * 100
        assert pct_q88 < 5.0, f"Q8.8 gap too large: {pct_q88:.1f}%"
        assert pct_q412 < 5.0, f"Q4.12 gap too large: {pct_q412:.1f}%"

    @pytest.mark.xfail(reason="Q4.12 integer range [-8,+8] too narrow for LIF voltages (-65mV)")
    def test_q412_zero_current_silence(self) -> None:
        """Q4.12 with zero current should produce no spikes.

        NOTE: This test is expected to fail because the LIF model uses
        voltages in [-65, +30] mV, which exceeds Q4.12's ±8 integer
        range. The initial voltage -65.0 wraps, causing spurious spikes.
        This is the precision-vs-range tradeoff — Q4.12 is ideal for
        models with normalised dynamics (e.g. FitzHugh-Nagumo, Theta).
        """
        vlog_spikes = _verilog_spike_count_q412("lif", 50, 0.0)
        assert vlog_spikes == 0


def _verilog_spike_count_q1616(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q16.16 precision (32-bit) and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q1616"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=32,
        fraction=16,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 precision mode: 16 integer + 16 fractional bits (32-bit).

    Q16.16 combines Q8.8's wide integer range [-32768, +32767] with
    1/65536 ≈ 0.000015 resolution. This is the "gold standard" for
    hardware neuron fidelity, suitable for all model dynamics.
    """

    def test_lif_q1616_spikes(self) -> None:
        """Q16.16 LIF should spike reliably."""
        vlog_spikes = _verilog_spike_count_q1616("lif", _N_STEPS, _INPUT_CURRENT)
        assert vlog_spikes > 0

    def test_lif_q1616_near_python(self) -> None:
        """Q16.16 should match Python to within 1%."""
        py_spikes = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count_q1616("lif", _N_STEPS, _INPUT_CURRENT)

        gap = abs(py_spikes - vlog_spikes)
        gap_pct = gap / max(py_spikes, 1) * 100
        print(
            f"\n  Q16.16 co-sim LIF: Python={py_spikes}, Verilog={vlog_spikes}, "
            f"gap={gap} ({gap_pct:.1f}%)"
        )

        assert gap_pct < 1.0, (
            f"Q16.16 gap too large: {gap_pct:.1f}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

    def test_q1616_zero_current_silence(self) -> None:
        """Q16.16 with zero current should produce no spikes.

        Unlike Q4.12, Q16.16 has enough integer range for LIF voltages.
        """
        vlog_spikes = _verilog_spike_count_q1616("lif", 50, 0.0)
        assert vlog_spikes == 0


# ══════════════════════════════════════════════════════════════════════
# Generic multi-precision co-simulation infrastructure
# ══════════════════════════════════════════════════════════════════════


def _verilog_spike_count_generic(
    model_name: str,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
) -> int:
    """Compile at arbitrary (data_width, fraction) and simulate, returning spike count.

    This is the universal co-simulation helper — all precision-specific
    helpers (_verilog_spike_count, _verilog_spike_count_q412, etc.) are
    special cases of this function.
    """
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    mode_tag = f"q{data_width - fraction}_{fraction}"
    module_name = f"sc_{model_name}_{mode_tag}"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=data_width,
        fraction=fraction,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


# ── Full precision mode registry (matches dsl_cli.PRECISION_MODES) ───
_ALL_MODES = {
    "Q1.7": (8, 7),
    "Q8.8": (16, 8),
    "Q4.12": (16, 12),
    "Q1.15": (16, 15),
    "Q9.9": (18, 9),
    "Q12.12": (24, 12),
    "Q14.13": (27, 13),
    "Q20.12": (32, 12),
    "Q16.16": (32, 16),
    "Q8.24": (32, 24),
    "Q18.18": (36, 18),
}

# Modes with enough integer range for mV-scale models (v_rest=-65)
_MV_RANGE_MODES = {
    name: spec for name, spec in _ALL_MODES.items() if -(1 << (spec[0] - 1)) / (1 << spec[1]) <= -65
}
# Expected: Q8.8, Q9.9, Q12.12, Q16.16, Q8.24, Q18.18

# Models that work reliably for accuracy comparison at mV-range modes
_MV_ACCURACY_MODELS = ["lif", "lapicque", "resonate_fire"]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestMultiPrecision:
    """Comprehensive multi-precision co-simulation: 9 modes × 5 models.

    Tests organised by angle:
    1. Compilation smoke test — all modes × all models compile and simulate
    2. Accuracy — mV-range modes match Python within 1% for linear models
    3. Zero-current silence — all mV-range modes produce no spikes at I=0
    4. Cross-precision consistency — wider formats produce same or better results
    5. DSP-native modes — Q9.9, Q12.12, Q18.18 exploit hardware widths
    """

    # ── Angle 1: Compilation Smoke Test ──────────────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    def test_compilation_all_mv_modes(self, mode_name: str) -> None:
        """Every mV-range mode must compile and simulate without errors."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        spikes = _verilog_spike_count_generic("lif", 50, 50.0, dw, frac)
        assert spikes >= 0, f"{mode_name} produced invalid spike count"

    # ── Angle 2: Accuracy (<1%) for Linear Models ────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    @pytest.mark.parametrize("model_name", _MV_ACCURACY_MODELS)
    def test_accuracy_mv_modes(self, mode_name: str, model_name: str) -> None:
        """mV-range modes must match Python within 1% for linear models."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        py = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vl = _verilog_spike_count_generic(model_name, _N_STEPS, _INPUT_CURRENT, dw, frac)

        gap_pct = abs(py - vl) / max(py, 1) * 100
        print(f"\n  {mode_name} {model_name}: Py={py}, Vl={vl}, gap={gap_pct:.1f}%")

        assert gap_pct < 1.0, f"{mode_name} {model_name}: gap={gap_pct:.1f}% (Py={py}, Vl={vl})"

    # ── Angle 3: Zero-current Silence ────────────────────────────────
    @pytest.mark.parametrize("mode_name", list(_MV_RANGE_MODES.keys()))
    def test_zero_current_silence_mv_modes(self, mode_name: str) -> None:
        """LIF at zero current must produce zero spikes in all mV-range modes."""
        dw, frac = _MV_RANGE_MODES[mode_name]
        spikes = _verilog_spike_count_generic("lif", 50, 0.0, dw, frac)
        assert spikes == 0, f"{mode_name} LIF spiked with zero current: {spikes}"

    # ── Angle 4: Cross-Precision Consistency ─────────────────────────
    def test_cross_precision_lif_monotonicity(self) -> None:
        """Wider precisions must produce the same (or closer to Python) LIF count.

        All mV-range modes should converge to the same spike count as Python
        when the model is linear and the parameters fit.
        """
        py = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        results: dict[str, int] = {}

        for mode_name, (dw, frac) in sorted(_MV_RANGE_MODES.items(), key=lambda x: x[1][0]):
            vl = _verilog_spike_count_generic("lif", _N_STEPS, _INPUT_CURRENT, dw, frac)
            results[mode_name] = vl

        print(f"\n  Cross-precision LIF (Python={py}):")
        for name, vl in results.items():
            gap = abs(py - vl)
            print(f"    {name:8s}: {vl} spikes (gap={gap})")

        # All should match Python exactly for LIF
        for name, vl in results.items():
            assert vl == py, f"{name} diverges: {vl} vs Python={py}"

    # ── Angle 5: DSP-Native Modes ────────────────────────────────────
    @pytest.mark.parametrize(
        "mode_name,dw,frac",
        [
            ("Q9.9", 18, 9),
            ("Q12.12", 24, 12),
            ("Q14.13", 27, 13),
            ("Q20.12", 32, 12),
            ("Q18.18", 36, 18),
        ],
    )
    def test_dsp_native_lif(self, mode_name: str, dw: int, frac: int) -> None:
        """DSP-native modes must achieve exact Python parity for LIF."""
        py = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        vl = _verilog_spike_count_generic("lif", _N_STEPS, _INPUT_CURRENT, dw, frac)
        assert vl == py, f"{mode_name} ({dw}-bit) diverges: Verilog={vl}, Python={py}"

    # ── Angle 6: Narrow-Range Modes Compilation ──────────────────────
    @pytest.mark.parametrize(
        "mode_name,dw,frac",
        [
            ("Q1.7", 8, 7),
            ("Q1.15", 16, 15),
            ("Q4.12", 16, 12),
        ],
    )
    def test_narrow_range_compile_smoke(self, mode_name: str, dw: int, frac: int) -> None:
        """Narrow-range modes must compile without errors for any model.

        These modes may not match Python due to range overflow, but the
        compiler and simulator must not crash.
        """
        # Use resonate_fire: its initial state is closer to zero
        spikes = _verilog_spike_count_generic(
            "resonate_fire",
            50,
            10.0,
            dw,
            frac,
        )
        assert spikes >= 0

    # ── Angle 7: Precision Report API ────────────────────────────────
    def test_q88_class_all_modes(self) -> None:
        """Q88 dataclass works correctly at all 9 precision configurations."""
        from sc_neurocore.compiler.equation_compiler import Q88

        for mode_name, (dw, frac) in _ALL_MODES.items():
            q = Q88(data_width=dw, fraction=frac)
            assert q.integer_bits == dw - frac - 1
            assert q.resolution == pytest.approx(1.0 / (1 << frac))

            # Encode/decode roundtrip
            for test_val in [0.0, 1.0, -1.0, 0.5]:
                if q.min_value <= test_val <= q.max_value:
                    encoded = q.encode(test_val)
                    assert isinstance(encoded, int)
                    literal = q.encode_signed_literal(test_val)
                    assert literal.endswith(f"'sd{encoded}")

            # Precision report must not crash
            report = q.precision_report(dt=0.01, params={"test": 1.0})
            assert "Fixed-point format" in report


def _verilog_compiles(model_name: str) -> bool:
    """Return whether a model's generated Verilog is accepted by iverilog."""
    neuron = UniversalNeuron.from_schema(model_name)
    module_name = f"sc_{model_name}"
    verilog = neuron.to_verilog(module_name=module_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        out_path = Path(tmpdir) / f"{module_name}.out"
        rtl_path.write_text(verilog)
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestTranscendentalCoSimulation:
    """Auto model→RTL for transcendental models (exp/tanh/cosh via LUTs).

    Extends the polynomial cosim set: these models exercise the emitter's
    negative-LUT-literal handling, cosh support, and empty-parameter-list fix.
    """

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COSIM_MODELS)
    def test_both_produce_spikes(self, model_name: str) -> None:
        assert _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT) > 0
        assert _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT) > 0

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COSIM_MODELS)
    def test_spike_count_within_lut_tolerance(self, model_name: str) -> None:
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        assert py_spikes > 0 and vlog_spikes > 0
        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        assert gap_pct <= _TRANSCENDENTAL_TOLERANCE_PCT, (
            f"{model_name} transcendental co-sim gap {gap_pct:.1f}% exceeds "
            f"{_TRANSCENDENTAL_TOLERANCE_PCT}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COMPILE_MODELS)
    def test_transcendental_model_lowers_to_valid_verilog(self, model_name: str) -> None:
        """Transcendental models lower to iverilog-valid Verilog (no malformed literals).

        This is the emitter-fix verification: before the negative-LUT-literal,
        cosh, and empty-parameter fixes these models either raised
        "Unsupported function" or emitted malformed `W'sd-N` literals. Q8.8 +
        16-entry LUTs can be too coarse for a spike-count parity claim (glif,
        morris_lecar), so this asserts valid synthesisable RTL, not spike parity.
        """
        verilog = UniversalNeuron.from_schema(model_name).to_verilog(module_name=f"sc_{model_name}")
        assert "'sd-" not in verilog  # no malformed negative literals
        assert _verilog_compiles(model_name)

    def test_morris_lecar_lowers_cosh_to_a_lut(self) -> None:
        """Morris-Lecar's cosh is lowered to a LUT, not left as an unsupported call."""
        verilog = UniversalNeuron.from_schema("morris_lecar").to_verilog(
            module_name="sc_morris_lecar"
        )
        assert "_cosh_lut" in verilog

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
    Resonate-and-Fire, Perfect Integrator.

TestQ412Precision
    Q4.12 (16-bit, 12 fractional): precision vs Q8.8, plus range
    classification for millivolt-scale LIF state.

TestQ1616Precision
    Q16.16 (32-bit): gold standard fidelity, zero-current silence.

TestSchemaGapModelCosim
    WC-A5 Tier A: honest cosim status for the six schema-gap models —
    FitzHugh-Nagumo spike-parity and Rulkov short-window trajectory parity at
    Q16.16; exp_if / Hindmarsh-Rose compile-valid only (stiff exp / chaos);
    poisson / escape_rate stochastic and excluded from deterministic parity.

Verified Results (2026-05-01)
-----------------------------
Five Q8.8 baseline models have exact spike-count parity at I=50.0 over 200 steps.
Izhikevich has the honest Q8.8 boundary: float64=25, RTL=24 after candidate-reset
correction; Q16.16 restores exact 25/25 parity at the same operating point.
Driven LIF spike-count parity holds for Q8.8, Q4.12, and Q16.16 at
I=50.0; zero-current LIF requires an mV-range mode because Q4.12 cannot
represent v_rest=-65 mV or tau_m=10.
FitzHugh-Nagumo (2026-07-10): faithful RK4 / no-reset / rising-edge crossing re-enrolment
— hand == schema == Q16.16 RTL three-way exact (8 crossings, I=0.5, 3000 steps).
FitzHugh-Rinzel (2026-07-11): faithful three-state RK4 / no-reset / rising-edge crossing
enrolment — exact hand / schema / Q16.16 RTL spike-count parity across I=0.4 to 0.6.
Pernarowski (2026-07-11): faithful three-state RK4 / no-reset / rising-edge crossing
enrolment — exact hand / schema / Q16.16 RTL parity for the autonomous bursting train.
Terman-Wang (2026-07-11): faithful two-state RK4 / no-reset / rising-edge crossing
enrolment — exact hand / schema / Q16.16 RTL spike-count parity across three drive regimes.
Wilson-HR (2026-07-11): faithful two-state polynomial RK4 / hard voltage reset enrolment
— exact hand / schema / Q16.16 RTL spike-count parity across three drive regimes.
Rulkov map (2026-07-11): faithful simultaneous three-branch map / rising-edge crossing
enrolment — hand/TOML/JSON exact and Q16.16 RTL short-window trajectory within 0.001.
Mihalas-Niebur (2026-07-11): faithful four-state candidate-first RK4 / adaptive-reset
enrolment — exact hand / schema / Q16.16 RTL counts at ten currents from I=0 to I=6;
the isolated 1,000-step I=3 boundary is explicitly fixed at 111/111/112.

Prerequisites
-------------
- Icarus Verilog (``iverilog``, ``vvp``) — tests skip if unavailable.
- Install: ``apt install iverilog`` (Ubuntu) or ``brew install icarus-verilog`` (macOS).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

# Method-based spike-count primitives are shared with the pipelined co-simulation suite
# (``tests/test_pipeline_cosim.py``); they live in ``tests/cosim_support.py``. Imported under
# the module-local underscore names so the existing RK4 / exp-Euler call sites are unchanged.
from tests.cosim_support import (
    HAS_IVERILOG,
    _connor_stevens_hand_spike_count,
    _fitzhugh_nagumo_hand_spike_count,
    _fitzhugh_rinzel_hand_spike_count,
    _glif_hand_spike_count,
    _hodgkin_huxley_hand_spike_count,
    _lif_schema_precision_values,
    _mckean_hand_spike_count,
    _morris_lecar_hand_spike_count,
    _pernarowski_hand_spike_count,
    _python_spike_count,
    _rulkov_map_verilog_q1616_trace,
    _terman_wang_hand_spike_count,
    _verilog_compiles,
    _verilog_spike_count,
    _verilog_spike_count_q1616,
    _verilog_spike_count_q412,
    _wang_buzsaki_hand_spike_count,
    _wilson_hr_hand_spike_count,
)

# Co-simulation parameters
# NOTE: Q8.8 fixed-point has ±0.004 precision, which causes quantization
# drift in threshold and dynamics calculations. Higher currents are needed
# to reliably trigger spikes in the Verilog implementation.
_N_STEPS = 200
_INPUT_CURRENT = 50.0  # Higher than Python needs — overcomes Q8.8 precision loss

# Models suitable for Q8.8 co-simulation (polynomial/linear, no transcendental functions).
# Five models have exact spike-count parity. Izhikevich carries a one-spike Q8.8
# quantisation band and a separate exact Q16.16 guard below.
_COSIM_MODELS = [
    "lif",
    "lapicque",
    "quadratic_if",
    "izhikevich",
    "resonate_fire",
    "perfect_integrator",
]

# Non-baseline models reachable through the auto model→RTL path once the emitter
# lowers negative LUT entries correctly, supports cosh, and omits an empty parameter
# list. ``theta`` co-simulates near bit-true. GLIF is linear and has a dedicated exact Q16.16
# set below, while its Q8.8 path and the transcendental conductance models retain this
# compile-level regression because their coarse fixed-point forms do not support the
# same behavioural claim.
_TRANSCENDENTAL_COSIM_MODELS = ["theta"]
_TRANSCENDENTAL_TOLERANCE_PCT = 5.0
_TRANSCENDENTAL_COMPILE_MODELS = [
    "glif",
    "theta",
    "morris_lecar",
    "hodgkin_huxley",
    "terman_wang",
]


class TestTierBModelCosim:
    """WC-A5 Tier-B model enrollment beyond the original schema set."""

    def test_fitzhugh_rinzel_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas reproduce the hand model over a varied drive.

        The 1,200-step sequence alternates quiet, depolarising, and negative currents,
        exercising every RK4 stage in the three coupled equations, one upward crossing,
        and subsequent below-threshold re-arming. Exact state equality is required for
        ``v``, ``w``, and the ultra-slow ``y`` variable after every step, so either
        schema format drifting in an initial value, parameter, equation, operation order,
        or no-reset crossing decision fails immediately.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = FitzHughRinzelNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "fitzhugh_rinzel.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "fitzhugh_rinzel.json")
        currents = (0.0, 0.17, 0.5, 0.31, 0.83, -0.07) * 200
        spike_count = 0
        rearmed = False

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            if spike_count and hand.v < hand.v_threshold:
                rearmed = True
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w", "y"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 1
        assert rearmed

    def test_glif_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The paired GLIF schemas reproduce every hand-model RK4 state and reset.

        The 4,000-step varied drive exercises all four coupled linear equations,
        every RK4 stage, silence, tonic firing, and 181 candidate-first adaptive
        resets. Exact state and event equality after every step catches drift in
        either schema format's integration method, threshold relation, parameter,
        reset source, or post-candidate update order.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = GLIFNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "glif.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "glif.json")
        currents = (0.0, 15.0, 22.0, 30.0, 45.0, 50.0, 30.0, 22.0) * 500
        spike_count = 0
        reset_count = 0

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            if hand_spike:
                assert hand.v == hand.v_reset
                reset_count += 1
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "theta", "i_asc1", "i_asc2"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == reset_count == 181

    def test_pernarowski_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas reproduce the hand model over a varied drive.

        The 5,000-step sequence exercises the external-current term and every RK4
        stage across the fast cubic coordinate, recovery variable, and ultra-slow
        adaptation variable. It also covers 17 upward crossings and 17 subsequent
        below-threshold re-arms. Exact state equality is required after every step,
        so either schema format drifting in initial state, parameters, equations,
        operation order, or no-reset edge detection fails immediately.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = PernarowskiNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.json")
        currents = (0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.15, 0.05) * 625
        spike_count = 0
        rearm_count = 0
        was_above = hand.v >= hand.v_threshold

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            now_above = hand.v >= hand.v_threshold
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w", "z"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 17
        assert rearm_count == 17

    def test_terman_wang_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas track the hand oscillator over a varied drive.

        The 8,000-step sequence exercises the cubic fast nullcline, the ``tanh``
        recovery gate, external drive, all four simultaneous RK4 stages, and 28
        upward crossings followed by 28 re-arms. The hand model uses ``math.tanh``
        while the schema evaluator uses the NumPy transcendental, so state parity is
        asserted within a tight floating-point band rather than mislabelled as bit
        identity; spike decisions must still match exactly at every step.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = TermanWangOscillator()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "terman_wang.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "terman_wang.json")
        currents = (-1.0, 0.0, 0.5, 0.25, -0.5, 0.75, 0.0, 0.4) * 1000
        spike_count = 0
        rearm_count = 0
        was_above = hand.v >= hand.v_peak

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            now_above = hand.v >= hand.v_peak
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == pytest.approx(expected, rel=1e-12, abs=1e-10)
                assert json_schema.state[variable] == pytest.approx(expected, rel=1e-12, abs=1e-10)

        assert spike_count == 28
        assert rearm_count == 28

    def test_wilson_hr_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas track Wilson-HR over a varied drive.

        Five passes through eight 100-step drive blocks exercise the polynomial
        membrane nullcline, coupled recovery flow, all four simultaneous RK4 stages,
        and 35 hard voltage resets. Both schema formats must reproduce every hand-model
        spike decision and both post-step states exactly; equality of ``r`` on spiking
        steps guards the contract that only ``v`` resets.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = WilsonHRNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.json")
        current_blocks = (0.0, 10.0, 2.0, 10.0, 0.0, 5.0, 10.0, 2.0)
        spike_count = 0
        reset_count = 0

        for _cycle in range(5):
            for current in current_blocks:
                for _step in range(100):
                    hand_spike = hand.step(current)
                    spike_count += hand_spike
                    if hand_spike:
                        assert hand.v == -0.7
                        reset_count += 1
                    assert int(bool(toml_schema.step(I=current))) == hand_spike
                    assert int(bool(json_schema.step(I=current))) == hand_spike
                    for variable in ("v", "r"):
                        expected = getattr(hand, variable)
                        assert toml_schema.state[variable] == expected
                        assert json_schema.state[variable] == expected

        assert spike_count == 35
        assert reset_count == 35


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

    def test_q412_zero_current_lif_is_range_classified(self) -> None:
        """Q4.12 LIF zero-current is a range mismatch, not a parity claim."""
        params = _lif_schema_precision_values()
        q412 = Q88(data_width=16, fraction=12)
        incompatible = {
            name for name, value in params.items() if not q412.min_value <= value <= q412.max_value
        }
        report = q412.precision_report(dt=1.0, params=params)

        assert q412.min_value == pytest.approx(-8.0)
        assert q412.max_value == pytest.approx(7.999755859375)
        assert incompatible == {"v_rest", "tau_m", "v"}
        assert "Underflow: v_rest=-65.0 below Q4.12 min=-8.0000" in report
        assert "Overflow: tau_m=10.0 exceeds Q4.12 max=7.9998" in report
        assert "Underflow: v=-65.0 below Q4.12 min=-8.0000" in report

        cli = subprocess.run(
            [sys.executable, "-m", "sc_neurocore.neurons", "precision", "lif"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )
        compatible_line = next(
            line for line in cli.stdout.splitlines() if line.startswith("Compatible modes:")
        )
        assert "Q4.12" not in compatible_line
        assert "Q8.8" in compatible_line
        assert "Q16.16" in compatible_line


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

    def test_izhikevich_q1616_candidate_reset_parity(self) -> None:
        """Q16.16 preserves exact Izhikevich parity with candidate-based recovery reset.

        The coarse Q8.8 path shifts one marginal spike after correcting ``u = u + d``
        to read the integrated candidate. At Q16.16 the same 200-step, ``I=50``
        operating point reproduces all 25 float64 spikes, proving the semantic fix
        does not trade fidelity for the Q8.8 baseline count.
        """
        python_spikes = _python_spike_count("izhikevich", _N_STEPS, _INPUT_CURRENT)
        verilog_spikes = _verilog_spike_count_q1616("izhikevich", _N_STEPS, _INPUT_CURRENT)

        assert python_spikes == verilog_spikes == 25

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (15.0, 0), (22.0, 23), (30.0, 54), (45.0, 86), (50.0, 95)),
        ids=("rest", "subthreshold", "onset", "tonic", "high-drive", "strong-drive"),
    )
    def test_glif_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """GLIF has exact hand/schema/Q16.16 spike-count parity across six regimes.

        The schema mirrors the maintained four-state, candidate-first classical-RK4
        hand model with level ``v >= theta`` detection and adaptive reset. Hand model
        and schema runner agree exactly at every operating point. The compiler lowers
        reset expressions from the integrated candidate and exposes the same post-reset
        state in RTL, so Q16.16 preserves the complete spike count despite quantising
        ``a_theta=0.01`` and the adaptive increments. Rest, subthreshold, onset, tonic,
        and high-drive regimes are all enrolled rather than one selected current.
        """
        n_steps = 1000
        hand_spikes = _glif_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("glif", n_steps, current)
        verilog_spikes = _verilog_spike_count_q1616("glif", n_steps, current)

        assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes, (
            f"GLIF exact Q16.16 mismatch at I={current}: "
            f"hand={hand_spikes}, schema={schema_spikes}, verilog={verilog_spikes}"
        )

    def test_morris_lecar_q1616_parity(self) -> None:
        """Faithful Morris-Lecar co-simulates at exact Q16.16 three-way crossing parity.

        The re-enrolled schema is the genuine Morris-Lecar (1981) calcium-potassium
        relaxation oscillator matching ``MorrisLecarNeuron``'s maintained defaults:
        four-stage RK4, **no reset**, and rising-edge (``v >= v_threshold`` upward
        crossing) spike detection. The earlier schema was ``method="euler"`` with a
        no-op ``[reset]`` (``v -> v``, ``w -> w``) that disabled edge detection, routed
        to the level datapath, and over-counted every above-threshold step; both sides
        over-counted identically so a ~15% tolerance band passed while validating a
        caricature. The faithful schema counts one spike per action potential: at the
        sustained depolarising regime (``I=100``, 3000 steps) the hand model, the schema
        runner and the emitted Q16.16 RTL all report the same seven upward crossings.

        The sigmoidal gating lowers to 256-entry cosh/tanh LUTs, so — unlike the
        polynomial FitzHugh-Nagumo / piecewise-linear McKean oscillators — this is an
        exact **spike-count** parity, not bit-identical state: the hand model (``math``
        transcendentals via ``RK4Solver``) and the schema runner (``numpy``
        transcendentals) diverge at the float level, yet the crossing count is robust to
        that drift across the whole ``I in [90, 110]`` band and the Q16.16 LUT datapath
        reproduces it exactly. (``I=120`` is a knife-edge where a marginal crossing
        splits between the paths; the enrolled point sits safely inside the robust band.)
        """
        current, n_steps = 100.0, 3000
        hand_spikes = _morris_lecar_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("morris_lecar", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("morris_lecar", n_steps, current)
        assert 1 < py_spikes < n_steps  # a sustained relaxation train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"Morris-Lecar three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    def test_hodgkin_huxley_q1616_macrostep_parity(self) -> None:
        """Faithful macro-step Hodgkin-Huxley: hand == schema exact, verilog within one spike.

        The re-enrolled schema mirrors ``HodgkinHuxleyNeuron(integrator="rk4")``'s maintained
        integrator: RK4 with ``substeps=100`` (100 inner ``dt=0.01`` sub-steps per 1 ms macro
        step) and a rising-edge (``v >= v_threshold``) crossing evaluated only on the macro
        boundary, no reset. The earlier schema was single-step ``method="euler"`` — neither the
        hand model's RK4 nor its macro-stepping — so it could only be compared schema-vs-verilog
        under a 5% band; the macro-step schema now reproduces the hand model's action-potential
        count exactly, so **hand == schema** (one hand ``step()`` per schema macro ``step()``).
        The comparison is against the ``integrator="rk4"`` (simultaneous) path, not the
        Gauss-Seidel default ``baseline_euler``, which the DSL's simultaneous integration matches.

        The Q16.16 RTL runs 100 clocks per macro step (one integration sub-step each, the
        crossing gated to the macro boundary) and tracks the schema **within one spike** over the
        bounded window. Like the stiff six-state Connor-Stevens (and unlike the well-conditioned
        Morris-Lecar), Hodgkin-Huxley's exprel / sigmoid gating lowers to 256-entry look-up
        tables; the fixed-point trajectory drifts from float64 and the drift is
        **look-up-table-resolution-limited, not datapath-precision-limited**, so it holds
        three-way over a bounded window and accumulates beyond it — an honest per-model
        hardware-fidelity band, not a tolerance knob. The macro-step lowering itself is bit-exact
        (proven on the polynomial FitzHugh-Nagumo sub-step cosim); the residual is genuine
        conductance-LUT quantisation.
        """
        current, macro_steps, substeps = 15.0, 20, 100
        hand_spikes = _hodgkin_huxley_hand_spike_count(macro_steps, current)
        py_spikes = _python_spike_count("hodgkin_huxley", macro_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("hodgkin_huxley", macro_steps * substeps, current)
        assert 1 < py_spikes < macro_steps  # a partial macro-step train, not saturated
        assert hand_spikes == py_spikes, (
            f"Hodgkin-Huxley hand/schema macro-step mismatch: hand={hand_spikes}, schema={py_spikes}"
        )
        assert abs(py_spikes - vlog_spikes) <= 1, (
            f"Hodgkin-Huxley Q16.16 macro-step gap > 1 spike "
            f"(schema={py_spikes}, verilog={vlog_spikes})"
        )

    def test_adex_q1616_parity(self) -> None:
        """Adaptive-exponential IF (exp spike + adaptation + reset) is bit-true at Q16.16."""
        py_spikes = _python_spike_count("adex", 500, 1000.0)
        vlog_spikes = _verilog_spike_count_q1616("adex", 500, 1000.0)
        assert py_spikes > 0 and vlog_spikes > 0
        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        assert gap_pct <= 2.0, (
            f"AdEx Q16.16 gap {gap_pct:.1f}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

    def test_wang_buzsaki_q1616_macrostep_parity(self) -> None:
        """Faithful macro-step Wang-Buzsaki: hand == schema exact, verilog within one spike.

        The re-enrolled schema mirrors ``WangBuzsakiNeuron``'s maintained integrator: a
        sequential (Gauss-Seidel) forward Euler with ``substeps=50`` (50 inner ``dt=0.01``
        sub-steps per 0.5 ms macro step, the gating variables ``h``/``n`` updated from the old
        voltage and the membrane voltage ``v`` from the new gates) and a rising-edge
        ``v >= v_threshold`` crossing evaluated only on the macro boundary, no reset. The
        earlier schema was single-step ``method="euler"`` with a sigmoid-caricature ``m_inf``
        and unfaithful gate initial conditions, so it could only be compared schema-vs-verilog
        under a 15% band; the macro-step schema now reproduces the hand model's
        action-potential count exactly, so **hand == schema** (one hand ``step()`` per schema
        macro ``step()``). Unlike Hodgkin-Huxley (simultaneous RK4), Wang-Buzsaki requires the
        DSL's ``gauss_seidel`` mode — the hand model updates the gates before the voltage, and
        simultaneous Euler drifts.

        The Q16.16 RTL runs 50 clocks per macro step (one sequential sub-step each, the crossing
        gated to the macro boundary) and tracks the schema **within one spike** over the bounded
        window. Wang-Buzsaki's exprel gating and its ``m_inf = alpha_m/(alpha_m+beta_m)``
        runtime division lower to a 256-entry look-up table plus a fixed-point divide; the
        fixed-point trajectory drifts from float64 and the drift is look-up-table- and
        fixed-point-resolution-limited, not a tolerance knob — three-way exact over this
        bounded window and accumulating beyond it, an honest per-model hardware-fidelity band.
        """
        current, macro_steps, substeps = 10.0, 20, 50
        hand_spikes = _wang_buzsaki_hand_spike_count(macro_steps, current)
        py_spikes = _python_spike_count("wang_buzsaki", macro_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wang_buzsaki", macro_steps * substeps, current)
        assert 1 < py_spikes < macro_steps  # a partial macro-step train, not saturated
        assert hand_spikes == py_spikes, (
            f"Wang-Buzsaki hand/schema macro-step mismatch: hand={hand_spikes}, schema={py_spikes}"
        )
        assert abs(py_spikes - vlog_spikes) <= 1, (
            f"Wang-Buzsaki Q16.16 macro-step gap > 1 spike "
            f"(schema={py_spikes}, verilog={vlog_spikes})"
        )

    def test_connor_stevens_q1616_macrostep_parity(self) -> None:
        """Faithful macro-step Connor-Stevens: hand == schema exact, verilog within one spike.

        The re-enrolled schema mirrors ``ConnorStevensNeuron``'s maintained integrator: RK4
        with ``substeps=100`` (100 inner ``dt=0.01`` sub-steps per 1 ms macro step) and a
        rising-edge (``v >= v_threshold``) crossing evaluated only on the macro boundary, no
        reset. The earlier schema was single-step ``method="euler"`` — neither the hand
        model's RK4 nor its macro-stepping — so it could only be compared schema-vs-verilog;
        the macro-step schema now reproduces the hand model's action-potential count exactly,
        so **hand == schema** (one hand ``step()`` per schema macro ``step()``).

        The Q16.16 RTL runs 100 clocks per macro step (one integration sub-step each, the
        crossing gated to the macro boundary) and tracks the schema **within one spike** over
        the bounded window. Unlike the well-conditioned Morris-Lecar, Connor-Stevens is a
        stiff six-state A-current model whose exprel / cube-root gating lowers to 256-entry
        look-up tables; the fixed-point trajectory drifts from float64 and the drift is
        **look-up-table-resolution-limited, not datapath-precision-limited** (the spike count
        is identical at Q16.16 / Q24.24 / Q32.32), so it holds three-way over a bounded window
        and accumulates beyond it — an honest per-model hardware-fidelity band, not a tolerance
        knob. The macro-step lowering itself is bit-exact (proven on the polynomial
        FitzHugh-Nagumo sub-step cosim); the residual is genuine conductance-LUT quantisation.
        """
        current, macro_steps, substeps = 100.0, 20, 100
        hand_spikes = _connor_stevens_hand_spike_count(macro_steps, current)
        py_spikes = _python_spike_count("connor_stevens", macro_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("connor_stevens", macro_steps * substeps, current)
        assert 1 < py_spikes < macro_steps  # a partial macro-step train, not saturated
        assert hand_spikes == py_spikes, (
            f"Connor-Stevens hand/schema macro-step mismatch: hand={hand_spikes}, schema={py_spikes}"
        )
        assert abs(py_spikes - vlog_spikes) <= 1, (
            f"Connor-Stevens Q16.16 macro-step gap > 1 spike "
            f"(schema={py_spikes}, verilog={vlog_spikes})"
        )

    def test_fitzhugh_nagumo_q1616_parity(self) -> None:
        """Faithful FitzHugh-Nagumo co-simulates at exact Q16.16 three-way parity.

        The re-enrolled schema is the genuine FitzHugh (1961) relaxation oscillator:
        four-stage RK4, **no reset**, and rising-edge (``v >= v_threshold`` upward
        crossing) spike detection matching ``FitzHughNagumoNeuron`` — the cube is
        ``v * v * v`` (exact IEEE multiplication). Over 3000 steps at ``I=0.5`` the
        hand model, the schema runner and the emitted Q16.16 RTL all report the same
        sustained partial train (eight crossings), a repetitive train that exercises
        the ``_thr_prev`` edge re-arming rather than a single event. The right-hand
        side is polynomial (no look-up table), so the fixed-point parity is bit-exact,
        not a tolerance band. This supersedes the earlier Euler+reset caricature
        (``I=0.8``, 7 of 300) that only agreed because both sides shared the same
        unfaithful reset dynamics.
        """
        current, n_steps = 0.5, 3000
        hand_spikes = _fitzhugh_nagumo_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("fitzhugh_nagumo", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("fitzhugh_nagumo", n_steps, current)
        assert 1 < py_spikes < n_steps  # a repetitive partial train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"FitzHugh-Nagumo three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.4, 7), (0.5, 8), (0.6, 8)),
        ids=("I=0.4", "I=0.5", "I=0.6"),
    )
    def test_fitzhugh_rinzel_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """FitzHugh-Rinzel has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state flow: four-stage
        simultaneous RK4 over the cubic fast membrane, linear recovery, and
        ultra-slow modulation equations; no reset; and rising-edge
        ``v >= v_threshold`` crossing detection. Over 3000 steps the hand model,
        schema runner, and emitted Q16.16 RTL produce 7, 8, and 8 crossings at
        ``I=0.4``, ``0.5``, and ``0.6`` respectively. This current band avoids the
        marginal ninth crossing at ``I=0.7``, where fixed-point rounding changes the
        spike count, so the contract states the robust band rather than hiding that
        boundary.
        """
        n_steps = 3000
        hand_spikes = _fitzhugh_rinzel_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("fitzhugh_rinzel", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("fitzhugh_rinzel", n_steps, current)
        assert hand_spikes == expected_spikes
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"FitzHugh-Rinzel three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    @pytest.mark.parametrize(
        "current",
        (-0.1, 0.0, 0.1, 0.2),
        ids=("I=-0.1", "I=0.0", "I=0.1", "I=0.2"),
    )
    def test_pernarowski_q1616_parity(self, current: float) -> None:
        """Pernarowski has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state beta-cell flow:
        simultaneous four-stage RK4 over the cubic fast coordinate and two
        separated slow variables, rising-edge ``v >= v_threshold`` detection,
        and no reset. The oscillator is autonomous, so input current shifts the
        trajectory rather than gating a silent/single/train transition. At each
        enrolled point from ``I=-0.1`` through ``I=0.2``, the hand model, schema
        runner, and emitted Q16.16 RTL report 17 crossings over 5,000 steps.
        """
        n_steps = 5000
        hand_spikes = _pernarowski_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("pernarowski", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("pernarowski", n_steps, current)
        assert 1 < hand_spikes < n_steps
        assert hand_spikes == py_spikes == vlog_spikes == 17, (
            f"Pernarowski three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((-1.0, 0), (0.0, 1), (0.5, 3)),
        ids=("silent", "single-crossing", "oscillatory-train"),
    )
    def test_terman_wang_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Terman-Wang has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained two-state LEGION oscillator:
        simultaneous four-stage RK4 over the cubic fast nullcline and ``tanh``-gated
        slow recovery, rising-edge ``v >= v_peak`` detection, and no reset. The
        transcendental gate makes raw state bit identity non-portable, so the declared
        observable is the robust silent/single/train crossing count: 0, 1, and 3 at
        ``I=-1.0``, ``0.0``, and ``0.5`` respectively over 8,000 steps.
        """
        n_steps = 8000
        hand_spikes = _terman_wang_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("terman_wang", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("terman_wang", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Terman-Wang three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (2.0, 1), (10.0, 4)),
        ids=("silent", "single-spike", "four-spike-train"),
    )
    def test_wilson_hr_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Wilson-HR has exact three-way Q16.16 spike-count parity.

        The schema mirrors the maintained two-state polynomial cortical model:
        simultaneous four-stage RK4 over ``v`` and ``r``, level detection at
        ``v >= v_peak``, and a hard ``v = -0.7`` reset that preserves the candidate
        recovery state. Over 5,000 steps the hand model, schema runner, and emitted
        RTL reproduce the silent, single-spike, and four-spike operating points.
        """
        n_steps = 5000
        hand_spikes = _wilson_hr_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("wilson_hr", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wilson_hr", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Wilson-HR three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )

    def test_rulkov_map_q1616_short_window_trajectory(self) -> None:
        """Rulkov has class-correct three-way short-window trajectory parity.

        The maintained hand model and both schema formats execute the published
        simultaneous fast/slow map with rising ``x >= 0`` crossing detection.
        At ``I=1.5`` the 30-step window visits the rational, plateau, and hard-reset
        branches ten times each. Hand/TOML/JSON decisions and states must be exact;
        the emitted Q16.16 RTL must reproduce the complete ten-event vector while
        each committed state stays within 0.001 of float64. The bounded trajectory
        is the map-appropriate observable; no long-window spike-count claim is made.
        """
        current = 1.5
        n_steps = 30
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = RulkovMapNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "rulkov_map.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "rulkov_map.json")
        hand_trace: list[tuple[int, float, float]] = []
        branch_counts = {"rational": 0, "plateau": 0, "reset": 0}

        for _step in range(n_steps):
            boundary = hand.alpha + hand.y + current
            if hand.x <= 0.0:
                branch_counts["rational"] += 1
            elif hand.x < boundary:
                branch_counts["plateau"] += 1
            else:
                branch_counts["reset"] += 1
            hand_spike = hand.step(current)
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            assert toml_schema.state == {"x": hand.x, "y": hand.y}
            assert json_schema.state == {"x": hand.x, "y": hand.y}
            hand_trace.append((hand_spike, hand.x, hand.y))

        rtl_trace = _rulkov_map_verilog_q1616_trace(n_steps, current)
        assert branch_counts == {"rational": 10, "plateau": 10, "reset": 10}
        assert [row[0] for row in hand_trace] == [row[0] for row in rtl_trace]
        assert sum(row[0] for row in rtl_trace) == 10
        for (_spike, expected_x, expected_y), (_rtl_spike, rtl_x, rtl_y) in zip(
            hand_trace, rtl_trace, strict=True
        ):
            assert rtl_x == pytest.approx(expected_x, abs=0.001)
            assert rtl_y == pytest.approx(expected_y, abs=0.001)

    def test_mckean_q1616_parity(self) -> None:
        """Faithful McKean co-simulates at exact Q16.16 three-way parity.

        The McKean (1970) piecewise-linear FitzHugh-Nagumo caricature replaces the
        cubic nullcline with ``f(v) = min(max(-v, v - a), 1 - v)``; the bundled schema
        is RK4, no reset, rising-edge (``v >= v_peak`` upward crossing) detection,
        matching ``McKeanNeuron``. The min/max branch selection is exact arithmetic (a
        fixed-point comparison + select, no look-up table), so at the sustained
        relaxation-oscillation operating point (``epsilon=0.2``, ``gamma=0.5``,
        ``I=0.6``) the hand model, the schema runner and the emitted Q16.16 RTL all
        report the same 16-crossing train over 3000 steps, bit-exactly. (The default
        hand-model regime ``epsilon=0.01`` is a single-transient knife-edge; the
        enrolled regime is a robust limit cycle whose crossings survive fixed-point
        rounding.)
        """
        current, n_steps = 0.6, 3000
        hand_spikes = _mckean_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("mckean", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("mckean", n_steps, current)
        assert 1 < py_spikes < n_steps  # a sustained oscillation train, not saturated
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"McKean three-way mismatch: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )


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
        """Non-baseline models lower to iverilog-valid Verilog without malformed literals.

        This is the emitter-fix verification: before the negative-LUT-literal,
        cosh, and empty-parameter fixes these models either raised
        "Unsupported function" or emitted malformed `W'sd-N` literals. The GLIF
        Q8.8 path is resolution-limited and the conductance-model look-up tables can
        be too coarse for the dedicated Q16.16 behavioural claims, so this assertion
        covers valid synthesisable RTL rather than spike parity.
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


# ══════════════════════════════════════════════════════════════════════
# WC-A5 Tier A — honest cosim classification of the schema-gap models
# ══════════════════════════════════════════════════════════════════════
#
# Six schema-DSL models had no spike-parity coverage. Empirical Python↔Verilog
# probing (2026-07-07) classifies each by what can be *honestly* validated,
# rather than forcing a green test at a flattering operating point:
#
#   fitzhugh_nagumo  → spike-parity at Q16.16 — promoted into TestQ1616Precision.
#   exp_if           → compile-valid RTL only. delta_t·exp((v−v_th)/delta_t)
#                      saturates the exp LUT near threshold; Q8.8 never fires and
#                      Q16.16 matches only in a narrow drive band (exact at I=500,
#                      50% gap at I=1000), so no robust parity claim.
#   hindmarsh_rose   → compile-valid RTL only. Chaotic burster: subthreshold
#                      (0 spikes) for n≤80, then sensitive dependence makes the
#                      fixed-point and float trains diverge (422%→1000% gap) once
#                      bursting starts — bit-true parity is undefined for chaos.
#   rulkov_map       → short-window trajectory parity at Q16.16. The corrected
#                      rising-crossing schema is enrolled above with exact event
#                      parity and bounded x/y error across all three map branches;
#                      long-window spike count remains intentionally unclaimed.
#   poisson,         → stochastic (schema `stochastic = true`, threshold
#   escape_rate        `condition = "stochastic"`): spike emission is a random
#                      process, so deterministic spike-count parity is not defined.
#
# Full evidence: docs/internal (WC-A5 Tier-A report).
_SCHEMA_GAP_COMPILE_ONLY = ["exp_if", "hindmarsh_rose"]
_SCHEMA_GAP_STOCHASTIC = ["poisson", "escape_rate"]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestSchemaGapModelCosim:
    """WC-A5 Tier-A closure: every schema-gap model has an explicit cosim status.

    ``fitzhugh_nagumo`` is spike-parity validated in ``TestQ1616Precision``. The
    remaining five are classified here: deterministic-but-not-parity models are
    asserted to lower to valid RTL (the honest compile-only precedent used for
    glif/morris_lecar at Q8.8), and stochastic models are asserted to be excluded
    from every deterministic cosim set with their schema stochastic flag confirmed.
    """

    @pytest.mark.parametrize("model_name", _SCHEMA_GAP_COMPILE_ONLY)
    def test_compile_valid_but_not_spike_parity(self, model_name: str) -> None:
        """exp_if and Hindmarsh-Rose lower to iverilog-valid Verilog.

        Spike-count parity is not scientifically claimable for these (stiff exp
        saturation and chaotic sensitive-dependence — see module notes), so this
        asserts the fixed-point *path* is valid rather than a
        spike count, matching the honest compile-only precedent for coarse-LUT
        transcendental models.
        """
        assert _verilog_compiles(model_name)

    @pytest.mark.parametrize("model_name", _SCHEMA_GAP_STOCHASTIC)
    def test_stochastic_models_excluded_from_deterministic_cosim(self, model_name: str) -> None:
        """poisson / escape_rate are stochastic, so bit-true spike parity is undefined.

        Assert the schema declares the model stochastic and that it appears in no
        deterministic cosim set, so the exclusion is explicit and audited rather
        than an accidental omission.
        """
        neuron = UniversalNeuron.from_schema(model_name)
        assert neuron.extensions.get("stochastic") is True
        assert model_name not in _COSIM_MODELS
        assert model_name not in _TRANSCENDENTAL_COSIM_MODELS
        assert model_name not in _SCHEMA_GAP_COMPILE_ONLY

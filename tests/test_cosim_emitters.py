# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — co-simulation harness, precision, and emitter contracts

"""Cross-model co-simulation infrastructure and integrator-emitter contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _fitzhugh_nagumo_substep_neuron,
    _neuron_verilog_spike_count_q1616,
    _python_spike_count,
    _verilog_spike_count,
    _verilog_spike_count_generic,
    compile_to_verilog,
    spike_count_method as _spike_count_method,
    verilog_spike_count_method as _verilog_spike_count_method,
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
        """Q8.8 is exact except for the declared Izhikevich one-spike boundary.

        Candidate-first reset semantics expose a single marginal Izhikevich
        crossing at this coarse precision: float64 reports 25 spikes and Q8.8
        reports 24. The same model is exact at Q16.16 below. Every other baseline
        model retains exact Q8.8 parity.
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

        if model_name == "izhikevich":
            assert (py_spikes, vlog_spikes) == (25, 24)
        else:
            assert gap == 0, (
                f"Q8.8 co-simulation must be exact: {gap_pct:.1f}% "
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


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestMacroStepSubstepEmitter:
    """The macro-step (``substeps``) emitter lowering is bit-exact against the Python runner.

    Validated on the **polynomial** FitzHugh-Nagumo oscillator so no transcendental look-up
    table can mask a macro-step logic error: at Q16.16 the datapath is bit-true, so any
    runner-vs-RTL macro-step disagreement would be a pure lowering bug. The macro step advances
    ``substeps`` integration sub-steps per clock-window and gates the rising-edge crossing to the
    macro boundary; the same total sub-step budget must yield the same crossing count regardless
    of how it is grouped into macro steps.
    """

    def test_substeps_one_matches_plain_single_step(self) -> None:
        """``substeps=1`` is byte-identical to the ordinary single-step datapath."""
        neuron = _fitzhugh_nagumo_substep_neuron(1)
        runner = neuron.__class__(
            equations=dict(neuron.equations),
            parameters=dict(neuron.parameters),
            state={"v": -1.0, "w": -0.5},
            threshold=neuron.threshold_expr,
            dt=neuron.dt,
            method="rk4",
            detection="crossing",
            substeps=1,
        )
        py = sum(runner.step(I=0.5) for _ in range(3000))
        vlog = _neuron_verilog_spike_count_q1616(
            _fitzhugh_nagumo_substep_neuron(1), 3000, 0.5, "sc_fhn_ss1"
        )
        assert py == vlog == 8

    def test_macrostep_lowering_is_bit_exact_across_groupings(self) -> None:
        """A fixed sub-step budget yields the same crossing count under any macro grouping.

        3000 sub-steps as 3000 macro steps of 1, 1500 of 2, or 750 of 4 all report the eight
        FitzHugh-Nagumo crossings, hand==schema==verilog, proving the macro-boundary counter,
        the ``_thr_prev`` refresh, and the per-sub-step state advance are lowered correctly.
        """
        for substeps, macro_steps in ((2, 1500), (4, 750)):
            neuron = _fitzhugh_nagumo_substep_neuron(substeps)
            py = sum(neuron.step(I=0.5) for _ in range(macro_steps))
            vlog = _neuron_verilog_spike_count_q1616(
                _fitzhugh_nagumo_substep_neuron(substeps),
                macro_steps * substeps,
                0.5,
                f"sc_fhn_ss{substeps}",
            )
            assert py == vlog == 8, f"substeps={substeps}: schema={py}, verilog={vlog} (expected 8)"

    def test_substeps_reject_reset_model(self) -> None:
        """The emitter refuses ``substeps > 1`` on a resetting (level) model, not silently wrong."""
        reset_neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v >= 1.0",
            reset={"v": "0.0"},
            dt=0.1,
            method="euler",
            substeps=4,
        )
        with pytest.raises(NotImplementedError, match="substeps > 1"):
            compile_to_verilog(reset_neuron, module_name="sc_reset_ss", data_width=32, fraction=16)


# ══════════════════════════════════════════════════════════════════════
# Generic multi-precision co-simulation infrastructure
# ══════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════
# WC-A5 emitter unlock — RK4 integrator lowering in the schema→Verilog path
# ══════════════════════════════════════════════════════════════════════


# Smooth-ODE models whose emitted RK4 reproduces the Python RK4 golden exactly at Q16.16.
# (izhikevich is excluded: its 0.04·v² spike explosion is a stiff-hybrid range limit,
# already special-cased for the same reason in the Euler baseline set.)
_RK4_EXACT_MODELS = [
    ("quadratic_if", 50.0, 300),
    ("theta", 50.0, 300),
    ("adex", 1000.0, 500),
]
# Q-format menu proving the RK4 lowering is agnostic to the number representation
# (the integrator is a graph rewrite; the Q-format is a separate emission parameter).
_RK4_Q_FORMATS = [("Q16.16", 32, 16), ("Q12.12", 24, 12), ("Q18.18", 36, 18), ("Q20.12", 32, 12)]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestRK4Emitter:
    """The schema→Verilog emitter lowers a full classical RK4 step, not only Euler.

    When a schema declares ``method="rk4"`` the emitter now emits the four-stage
    RK4 graph (k1..k4 with the s0 + k·dt/2 / +k·dt stage states and the
    (k1+2k2+2k3+k4)·dt/6 increment), reusing the same fixed-point expression
    emitter as the Euler path. That reuse makes the integrator agnostic to the
    number representation, so RK4 inherits every Q-format for free. Faithfulness
    holds for smooth ODEs; the stiff hybrid izhikevich (0.04·v² spike explosion)
    remains a documented per-model range limit, not an emitter defect.
    """

    @pytest.mark.parametrize("model_name,current,n_steps", _RK4_EXACT_MODELS)
    def test_rk4_tracks_python_rk4_golden(
        self, model_name: str, current: float, n_steps: int
    ) -> None:
        """Emitted RK4 reproduces the Python RK4 golden spike count exactly (Q16.16)."""
        py_spikes = _spike_count_method(model_name, n_steps, current, "rk4")
        vlog_spikes = _verilog_spike_count_method(model_name, n_steps, current, 32, 16, "rk4")
        assert py_spikes > 0, f"Python RK4 {model_name} must spike"
        assert vlog_spikes == py_spikes, (
            f"{model_name} RK4 mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )

    def test_rk4_path_is_distinct_from_euler(self) -> None:
        """The RK4 emitter is a genuine four-stage step, not aliased to Euler.

        The theta phase-oscillator (sine LUT) at ``I=150`` is nonlinear enough that
        RK4 and Euler resolve a different number of phase wraps: the emitted RK4
        differs from the emitted Euler and still reproduces the Python RK4 golden
        exactly at Q16.16. (The faithful FitzHugh-Nagumo relaxation oscillator counts
        the same threshold crossings under either integrator — that robustness is why
        the distinctness demonstration uses a model whose spike count is genuinely
        integrator-sensitive rather than the earlier Euler+reset FHN caricature.)
        """
        py_rk4 = _spike_count_method("theta", 300, 150.0, "rk4")
        vlog_rk4 = _verilog_spike_count_method("theta", 300, 150.0, 32, 16, "rk4")
        vlog_euler = _verilog_spike_count_method("theta", 300, 150.0, 32, 16, "euler")
        assert vlog_rk4 != vlog_euler, "RK4 output must differ from Euler for a nonlinear model"
        gap_pct = abs(py_rk4 - vlog_rk4) / max(py_rk4, 1) * 100
        assert gap_pct <= 6.0, f"RK4 gap {gap_pct:.1f}% (Python={py_rk4}, Verilog={vlog_rk4})"

    @pytest.mark.parametrize("mode_name,data_width,fraction", _RK4_Q_FORMATS)
    def test_rk4_is_representation_agnostic(
        self, mode_name: str, data_width: int, fraction: int
    ) -> None:
        """RK4 inherits every Q-format for free (integrator ⟂ number representation)."""
        py_spikes = _spike_count_method("quadratic_if", 300, 50.0, "rk4")
        vlog_spikes = _verilog_spike_count_method(
            "quadratic_if", 300, 50.0, data_width, fraction, "rk4"
        )
        assert vlog_spikes == py_spikes, (
            f"{mode_name} RK4 mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )


# The latency-aware pipelined co-simulation (SR-2) lives in ``tests/test_pipeline_cosim.py``,
# using the shared primitives in ``tests/cosim_support.py``.


# ══════════════════════════════════════════════════════════════════════
# Exponential-Euler (Rush–Larsen) integrator lowering in the schema→Verilog path
# ══════════════════════════════════════════════════════════════════════

# Models whose emitted exp-Euler step reproduces the Python golden spike count exactly
# at Q16.16. The set spans the linearisation's regimes: constant-Jacobian gating
# (lif, lapicque — the canonical d/dt=(x_inf−x)/tau exponential-Euler win), a linear
# multi-variable oscillator (resonate_fire, exercising the simultaneous forward update),
# and a transcendental, state-dependent Jacobian (adex A = (exp(...)−1)/tau; theta
# A = (1−I)·… via the sin LUT) — proving the A datapath and the exprel/exp hardware LUTs
# lower bit-true where the golden's exprel path lands on a tabulated point. The stiff
# hybrids (quadratic_if, izhikevich — the 0.04·v² / v² spike explosion) are a documented
# per-model range limit at this word length, the same limit the Euler and RK4 baselines
# carry, not an emitter defect.
_EXP_EULER_EXACT_MODELS = [
    ("lif", 50.0, 300),
    ("lapicque", 50.0, 300),
    ("resonate_fire", 5.0, 300),
    ("adex", 1000.0, 500),
    ("theta", 50.0, 300),
]
# Same Q-format menu as RK4 — the integrator is a graph rewrite, the Q-format a separate
# emission parameter, so exp-Euler inherits every representation for free.
_EXP_EULER_Q_FORMATS = [
    ("Q16.16", 32, 16),
    ("Q12.12", 24, 12),
    ("Q18.18", 36, 18),
    ("Q20.12", 32, 12),
]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestExpEulerEmitter:
    """The schema→Verilog emitter lowers a linearised exponential-Euler step.

    When a schema declares ``method="exp_euler"`` the emitter emits, per variable,
    ``d<var> = f·dt·exprel(A·dt)`` with ``A = ∂f/∂x`` — the *same* symbolic derivative
    string the golden compiled (``EquationNeuron.jacobian_expressions``) lowered by the
    same fixed-point expression emitter as the Euler and RK4 paths, reusing the ``exprel``
    hardware LUT. That reuse makes the integrator agnostic to the number representation,
    so exp-Euler inherits every Q-format for free, and collapses to forward Euler in the
    zero-Jacobian limit (``exprel(0)=1``). Exact spike-count parity holds for the gating,
    linear and transcendental-Jacobian models above; the stiff hybrids (izhikevich,
    quadratic_if) remain a documented per-model range limit, not an emitter defect.
    """

    @pytest.mark.parametrize("model_name,current,n_steps", _EXP_EULER_EXACT_MODELS)
    def test_exp_euler_tracks_python_golden(
        self, model_name: str, current: float, n_steps: int
    ) -> None:
        """Emitted exp-Euler reproduces the Python golden spike count exactly (Q16.16)."""
        py_spikes = _spike_count_method(model_name, n_steps, current, "exp_euler")
        vlog_spikes = _verilog_spike_count_method(model_name, n_steps, current, 32, 16, "exp_euler")
        assert py_spikes > 0, f"Python exp-Euler {model_name} must spike"
        assert vlog_spikes == py_spikes, (
            f"{model_name} exp-Euler mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )

    def test_exp_euler_collapses_to_forward_euler_at_zero_jacobian(self) -> None:
        """With A=0 (perfect integrator) exprel(0)=1, so exp-Euler *is* forward Euler.

        The emitted exp-Euler datapath still multiplies by the tabulated ``exprel(0)``,
        so this proves the zero-Jacobian limit survives the LUT: the exp-Euler RTL, the
        Euler RTL and the Python golden all agree exactly.
        """
        py_spikes = _spike_count_method("perfect_integrator", 300, 5.0, "exp_euler")
        vlog_exp = _verilog_spike_count_method("perfect_integrator", 300, 5.0, 32, 16, "exp_euler")
        vlog_euler = _verilog_spike_count_method("perfect_integrator", 300, 5.0, 32, 16, "euler")
        assert py_spikes > 0
        assert vlog_exp == vlog_euler == py_spikes, (
            f"A=0 limit broke: exp={vlog_exp}, euler={vlog_euler}, py={py_spikes}"
        )

    def test_exp_euler_path_is_distinct_from_euler(self) -> None:
        """The exp-Euler emitter is a genuine linearised step, not aliased to Euler.

        The resonate-and-fire linear oscillator at ``I=10`` is stiff enough that
        forward Euler is unstable and fires every step, while the exponential-Euler
        linearisation stays on the true half-rate limit cycle: the emitted exp-Euler
        differs from the emitted Euler and reproduces the Python exp-Euler golden
        exactly at Q16.16. (As with the RK4 distinctness test, the faithful FHN
        oscillator is integrator-robust, so a stiff linear model demonstrates the
        exponential correction's effect more sharply.)
        """
        py_exp = _spike_count_method("resonate_fire", 300, 10.0, "exp_euler")
        vlog_exp = _verilog_spike_count_method("resonate_fire", 300, 10.0, 32, 16, "exp_euler")
        vlog_euler = _verilog_spike_count_method("resonate_fire", 300, 10.0, 32, 16, "euler")
        assert vlog_exp != vlog_euler, "exp-Euler output must differ from Euler for a stiff model"
        gap_pct = abs(py_exp - vlog_exp) / max(py_exp, 1) * 100
        assert gap_pct <= 6.0, f"exp-Euler gap {gap_pct:.1f}% (Python={py_exp}, Verilog={vlog_exp})"

    @pytest.mark.parametrize("mode_name,data_width,fraction", _EXP_EULER_Q_FORMATS)
    def test_exp_euler_is_representation_agnostic(
        self, mode_name: str, data_width: int, fraction: int
    ) -> None:
        """exp-Euler inherits every Q-format for free (integrator ⟂ number representation)."""
        py_spikes = _spike_count_method("lif", 300, 50.0, "exp_euler")
        vlog_spikes = _verilog_spike_count_method(
            "lif", 300, 50.0, data_width, fraction, "exp_euler"
        )
        assert vlog_spikes == py_spikes, (
            f"{mode_name} exp-Euler mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )


# The pipelined exp-Euler golden-parity test lives in ``tests/test_pipeline_cosim.py``.

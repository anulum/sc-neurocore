# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — edge-crossing threshold-detection capability

"""Tests for the ``detection = "crossing"`` rising-edge threshold capability.

A non-resetting oscillator (FitzHugh-Nagumo, McKean) spikes when its membrane crosses
threshold *upward*, not on every step the membrane stays above threshold. The schema DSL
now honours ``[threshold] detection = "crossing"`` in both the Python runner and the
emitted Verilog, so such oscillators co-simulate faithfully against their hand models.

Edge detection is engaged only for a crossing model that declares **no reset**: a reset
that drops the state below threshold already clears the condition every spike, so
``level`` and ``crossing`` are identical for reset-based integrate-and-fire models, which
keep the previously validated level datapath unchanged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import compile_to_verilog, generate_testbench
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

HAS_IVERILOG = shutil.which("iverilog") is not None

# --- Faithful in-memory schemas mirroring the hand oscillator models ------------------
# These are constructed via ``from_dict`` so the capability is validated end-to-end
# without disturbing the bundled ``fitzhugh_nagumo`` schema (whose faithful re-enrollment
# is a separate change with a wide test blast radius).

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


# --- Runner semantics -----------------------------------------------------------------


class TestEdgeDetectionRunner:
    """The Python runner's ``crossing`` vs ``level`` spike-decision semantics."""

    def test_crossing_matches_hand_oscillator_over_a_sequence(self) -> None:
        """A no-reset crossing schema reproduces the McKean hand model's edge decision."""
        hand = McKeanNeuron(dt=0.1, **_MCKEAN_INIT, **_MCKEAN_PARAMS)
        schema = UniversalNeuron.from_dict(_mckean_schema("crossing"))

        for current in (0.0, 0.2, 0.3, 0.2, 0.1, 0.2):
            for _ in range(200):
                assert int(bool(schema.step(I=current))) == hand.step(current)
                assert schema.state["v"] == hand.v
                assert schema.state["w"] == hand.w

    def test_level_over_counts_a_non_resetting_oscillator(self) -> None:
        """``level`` fires every step above threshold; ``crossing`` fires once per crossing.

        During a spike the membrane stays above ``v_peak`` for several steps, so a
        non-resetting oscillator run under ``level`` detection reports many more spikes
        than the true number of upward crossings — exactly the over-count that edge
        detection exists to prevent.
        """
        edge = UniversalNeuron.from_dict(_mckean_schema("crossing"))
        level = UniversalNeuron.from_dict(_mckean_schema("level"))
        edge_spikes = sum(1 for _ in range(1000) if edge.step(I=0.2))
        level_spikes = sum(1 for _ in range(1000) if level.step(I=0.2))

        assert edge_spikes > 0
        assert level_spikes > edge_spikes

    def test_edge_detection_flag_requires_crossing_and_no_reset(self) -> None:
        """``_edge_detection`` is set only for a crossing model that declares no reset."""
        crossing_no_reset = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_equation_neuron()
        assert crossing_no_reset._edge_detection is True

        level_no_reset = UniversalNeuron.from_dict(_fhn_schema("level")).to_equation_neuron()
        assert level_no_reset._edge_detection is False

        # A crossing model WITH a reset stays on the level path (reset clears the condition).
        crossing_with_reset = EquationNeuron(
            equations={"v": "-(v - E_L) / tau_m + I"},
            parameters={"E_L": -65.0, "tau_m": 10.0},
            state={"v": -65.0},
            threshold="v >= -50.0",
            reset={"v": "-65.0"},
            detection="crossing",
        )
        assert crossing_with_reset._edge_detection is False

    def test_reset_model_identical_under_level_and_crossing(self) -> None:
        """A reset-based integrate-and-fire model spikes identically either way."""

        def _lif(detection: str) -> EquationNeuron:
            return EquationNeuron(
                equations={"v": "-(v - E_L) / tau_m + I"},
                parameters={"E_L": -65.0, "tau_m": 10.0},
                state={"v": -65.0},
                threshold="v >= -50.0",
                reset={"v": "-65.0"},
                dt=1.0,
                detection=detection,
            )

        level_neuron = _lif("level")
        crossing_neuron = _lif("crossing")
        level_spikes = sum(level_neuron.step(I=5.0) for _ in range(200))
        crossing_spikes = sum(crossing_neuron.step(I=5.0) for _ in range(200))
        assert level_spikes > 0
        assert level_spikes == crossing_spikes

    def test_initial_threshold_active_seeds_edge_history(self) -> None:
        """The edge history is seeded from the initial state, not assumed ``False``."""
        below = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_equation_neuron()
        assert below.initial_threshold_active() is False
        assert below._prev_threshold_active is False

        # A no-reset crossing neuron that starts already above threshold seeds ``True`` so
        # it does not emit a spurious first-step spike before a genuine re-crossing.
        above = EquationNeuron(
            equations={"v": "-v + I", "w": "0.0 * w"},
            state={"v": 5.0, "w": 0.0},
            threshold="v >= 1.0",
            detection="crossing",
        )
        assert above.initial_threshold_active() is True
        assert above._prev_threshold_active is True
        assert above.step(I=0.0) == 0  # above threshold but no rising edge -> no spike

    def test_reset_reseeds_edge_history(self) -> None:
        """``reset()`` restores the seeded edge history so a re-run is reproducible."""
        neuron = UniversalNeuron.from_dict(_mckean_schema("crossing")).to_equation_neuron()
        for _ in range(300):
            neuron.step(I=0.2)
        neuron.reset()
        assert neuron._prev_threshold_active is neuron.initial_threshold_active()
        assert neuron.state == neuron.initial_state

    def test_no_threshold_neuron_never_edge_active(self) -> None:
        """A neuron without a threshold reports no initial activity and never edge-fires."""
        neuron = EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, detection="crossing")
        assert neuron.initial_threshold_active() is False
        assert neuron._edge_detection is False

    def test_invalid_detection_rejected(self) -> None:
        """An unknown detection mode fails closed rather than silently defaulting."""
        with pytest.raises(ValueError, match="detection must be one of"):
            EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, detection="edge")

    def test_previous_state_alias_exposes_unwrapped_candidate_crossing(self) -> None:
        """A backward phase wrap at negative current must not look like a spike."""
        schema = UniversalNeuron.from_dict(_wrapped_phase_schema(theta=0.01))

        assert schema.step(I=-0.5) == 0
        assert 0.0 <= schema.state["theta"] < 2.0 * 3.141592653589793
        assert schema.state["theta"] > 3.141592653589793

    def test_previous_state_alias_names_are_reserved(self) -> None:
        """A user parameter cannot shadow the generated macro-boundary alias."""
        with pytest.raises(ValueError, match="previous-state aliases"):
            EquationNeuron(
                equations={"v": "v + I"},
                parameters={"v_prev": 0.0},
                state={"v": 0.0},
                dt=1.0,
                method="map",
            )

    @pytest.mark.parametrize("detection", ["poisson", "escape_rate"])
    def test_stochastic_detection_markers_accepted_without_edge(self, detection: str) -> None:
        """Zero-probability stochastic trials advance RNG without level spikes."""
        probability_expression = "0.0" if detection == "poisson" else None
        rate_expression = "0.0" if detection == "escape_rate" else None
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="stochastic",
            detection=detection,
            probability_expression=probability_expression,
            rate_expression=rate_expression,
        )
        initial_rng = neuron.stochastic_rng_state

        assert [neuron.step(I=10.0) for _ in range(4)] == [0, 0, 0, 0]
        assert neuron.state["v"] == pytest.approx(4.0)
        assert neuron.stochastic_rng_state != initial_rng


# --- Verilog emitter ------------------------------------------------------------------


class TestEdgeDetectionEmitter:
    """The emitted RTL mirrors the runner's edge/level decision."""

    def test_crossing_no_reset_rtl_declares_edge_register(self) -> None:
        """A crossing, non-resetting model emits the 1-bit ``_thr_prev`` edge history."""
        rtl = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_verilog(module_name="fhn_edge")
        assert "reg _thr_prev;" in rtl
        assert "!_thr_prev" in rtl
        assert "_thr_prev <=" in rtl

    def test_level_model_rtl_has_no_edge_register(self) -> None:
        """A level model emits the unchanged datapath with no edge history register."""
        rtl = UniversalNeuron.from_dict(_fhn_schema("level")).to_verilog(module_name="fhn_level")
        assert "_thr_prev" not in rtl

    def test_reset_crossing_model_rtl_has_no_edge_register(self) -> None:
        """A crossing model with a reset stays on the level datapath (no edge register)."""
        neuron = EquationNeuron(
            equations={"v": "-(v - E_L) / tau_m + I"},
            parameters={"E_L": -65.0, "tau_m": 10.0},
            state={"v": -65.0},
            threshold="v >= -50.0",
            reset={"v": "-65.0"},
            detection="crossing",
        )
        rtl = compile_to_verilog(neuron, module_name="lif_reset_crossing")
        assert "_thr_prev" not in rtl

    def test_explicit_previous_state_crossing_reads_register_and_candidate(self) -> None:
        """The compiler maps ``theta_prev`` to the register, not wrapped next state."""
        rtl = UniversalNeuron.from_dict(_wrapped_phase_schema()).to_verilog(
            module_name="wrapped_phase"
        )

        assert "(theta_reg < P_THETA_THRESHOLD)" in rtl
        assert "P_THETA_THRESHOLD <= (theta_reg +" in rtl
        assert "_thr_prev" not in rtl


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestEdgeDetectionCosim:
    """End-to-end Python-runner vs emitted-RTL parity for edge-crossing oscillators."""

    def test_fitzhugh_nagumo_faithful_three_way_q1616_parity(self) -> None:
        """Faithful FitzHugh-Nagumo co-simulates at exact Q16.16 spike-count parity.

        The RK4, no-reset, crossing-detection schema reproduces ``FitzHughNagumoNeuron``
        bit-for-bit in float64, and the emitted Q16.16 RTL — using the new ``_thr_prev``
        rising-edge datapath — reproduces the same sustained relaxation-oscillation spike
        train exactly (8 of 3000 steps at ``I=0.5``), a partial train that exercises
        repeated crossing re-arming rather than a single event. The cubic right-hand side
        is polynomial, so no look-up table is involved and the parity is bit-exact.
        """
        current, n_steps = 0.5, 3000
        hand = _fhn_hand()
        hand_spikes = sum(hand.step(current) for _ in range(n_steps))

        schema = UniversalNeuron.from_dict(_fhn_schema("crossing"))
        schema_spikes = sum(1 for _ in range(n_steps) if schema.step(I=current))

        verilog_spikes = _verilog_spike_count_q1616(
            UniversalNeuron.from_dict(_fhn_schema("crossing")), n_steps, current, "fhn"
        )

        assert 1 < schema_spikes < n_steps  # a repetitive partial train
        assert hand_spikes == schema_spikes == verilog_spikes

    def test_faithful_fitzhugh_nagumo_schema_matches_hand_state_sequence(self) -> None:
        """The faithful schema mirrors the hand model's spike decision and both states."""
        hand = _fhn_hand()
        schema = UniversalNeuron.from_dict(_fhn_schema("crossing"))
        for current in (0.5, 0.5, 0.0, 0.6, 0.4):
            for _ in range(200):
                assert int(bool(schema.step(I=current))) == hand.step(current)
                assert schema.state["v"] == hand.v
                assert schema.state["w"] == hand.w

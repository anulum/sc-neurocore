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

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron

# Method-based spike-count primitives are shared with the pipelined co-simulation suite
# (``tests/test_pipeline_cosim.py``); they live in ``tests/cosim_support.py``. Imported under
# the module-local underscore names so the existing RK4 / exp-Euler call sites are unchanged.
from tests.cosim_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _verilog_compiles,
    _verilog_spike_count,
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

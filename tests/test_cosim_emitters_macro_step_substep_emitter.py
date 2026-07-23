# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMacroStepSubstepEmitter from former test_cosim_emitters.py

"""Focused suite: TestMacroStepSubstepEmitter from former test_cosim_emitters.py."""

from __future__ import annotations

from tests.cosim_emitters_support import *  # noqa: F403

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

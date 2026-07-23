# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEquivalenceProof from former test_equivalence_check.py

"""Focused suite: TestEquivalenceProof from former test_equivalence_check.py."""

from __future__ import annotations

from tests.equivalence_check_support import *  # noqa: F403

@_needs_formal
class TestEquivalenceProof:
    """End-to-end machine-checked proofs (require the formal toolchain)."""

    def test_equivalent_modules_are_proven(self, tmp_path: Path) -> None:
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            depth=10,
            workdir=tmp_path,
        )
        assert isinstance(result, EquivalenceResult)
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.returncode == 0
        assert result.counterexample is None

    def test_inequivalent_modules_are_disproved(self, tmp_path: Path) -> None:
        result = prove_equivalence(
            _TINY_DUT,
            _TINY_REF_BAD,
            _TINY_PORTS,
            dut_top="tiny_dut",
            ref_top="tiny_ref",
            depth=10,
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "FAIL"
        assert result.counterexample is not None
        assert "failed assertion" in result.counterexample.lower()
        assert result.trace_path is not None
        assert Path(result.trace_path).exists()

    def test_malformed_verilog_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="did not complete"):
            prove_equivalence(
                "module tiny_dut; this is not verilog",
                _TINY_REF,
                _TINY_PORTS,
                dut_top="tiny_dut",
                ref_top="tiny_ref",
                depth=4,
                workdir=tmp_path,
            )

    def test_generated_lif_matches_reference_model(self, tmp_path: Path) -> None:
        """The compiled LIF RTL primitive is equivalent to the reference model."""
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface

        ref_src = ref_path.read_text(encoding="utf-8")
        ports = parse_module_interface(ref_src, "sc_lif_reference", params={"DATA_WIDTH": 16})
        common = {"DATA_WIDTH": 16, "FRACTION": 8, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 256}
        result = prove_equivalence(
            dut_path.read_text(encoding="utf-8"),
            ref_src,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            depth=4,
            workdir=tmp_path,
        )
        assert result.proven is True

    def test_reference_threshold_mismatch_is_caught(self, tmp_path: Path) -> None:
        """A reference with the wrong threshold parameter must be disproved."""
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface

        ref_src = ref_path.read_text(encoding="utf-8")
        ports = parse_module_interface(ref_src, "sc_lif_reference", params={"DATA_WIDTH": 16})
        dut_params = {
            "DATA_WIDTH": 16,
            "FRACTION": 8,
            "V_REST": 0,
            "V_RESET": 0,
            "V_THRESHOLD": 256,
            "REFRACTORY_PERIOD": 0,
        }
        ref_params = {
            "DATA_WIDTH": 16,
            "FRACTION": 8,
            "V_REST": 0,
            "V_RESET": 0,
            "V_THRESHOLD": 512,
        }
        result = prove_equivalence(
            dut_path.read_text(encoding="utf-8"),
            ref_src,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params=dut_params,
            ref_params=ref_params,
            depth=8,
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "FAIL"

    def test_whitebox_taps_make_lif_provable_unbounded(self, tmp_path: Path) -> None:
        """Exposing internal state as taps lets k-induction prove the LIF unbounded.

        Naive k-induction on the miter is intractable (the fixed-point multiplier
        diverges from unreachable start states). Instrumenting both modules to
        expose the membrane register and refractory counter turns the miter's
        output-equality asserts into the state-matching invariant, which makes
        ``mode="prove"`` converge. A narrow 4-bit datapath keeps the multiplier
        tractable for the SMT solver.
        """
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut_wb = expose_state_taps(
            dut_path.read_text(encoding="utf-8"),
            top="sc_lif_neuron",
            taps=[
                StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "refractory_counter", msb="31"),
            ],
        )
        ref_wb = expose_state_taps(
            ref_path.read_text(encoding="utf-8"),
            top="sc_lif_reference",
            taps=[
                StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "32'd0", msb="31"),
            ],
        )
        common = {"DATA_WIDTH": 4, "FRACTION": 2, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 4}
        ports = parse_module_interface(ref_wb, "sc_lif_reference", params={"DATA_WIDTH": 4})
        result = prove_equivalence(
            dut_wb,
            ref_wb,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            mode="prove",
            depth=4,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_multiplier_abstraction_proves_lif_unbounded_full_width(self, tmp_path: Path) -> None:
        """Abstracting the multipliers lets k-induction prove the LIF at full width.

        Whitebox taps alone make k-induction *converge*, but bit-blasting the
        16-bit fixed-point multiplier keeps it intractable for the SMT solver.
        Lifting each product to a shared free input removes the multiplier from
        the solver entirely (the two instances see the same free product, so the
        abstraction is sound for a PASS), and the full 16-bit LIF proves unbounded.
        """
        dut_path = _REPO_ROOT / "hdl" / "sc_lif_neuron.v"
        ref_path = _REPO_ROOT / "hdl" / "equiv" / "sc_lif_reference.v"
        if not dut_path.exists() or not ref_path.exists():
            pytest.skip("committed LIF DUT / reference not present")
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            dut_path.read_text(encoding="utf-8"),
            top="sc_lif_neuron",
            signals=[
                LiftedSignal("leak_mul", "leak_product", msb="2*DATA_WIDTH-1", signed=True),
                LiftedSignal("in_mul", "input_product", msb="2*DATA_WIDTH-1", signed=True),
            ],
        )
        ref = abstract_to_free_inputs(
            ref_path.read_text(encoding="utf-8"),
            top="sc_lif_reference",
            signals=[
                LiftedSignal("leak_product", "leak_product", msb="2*DATA_WIDTH-1", signed=True),
                LiftedSignal("input_product", "input_product", msb="2*DATA_WIDTH-1", signed=True),
            ],
        )
        dut = expose_state_taps(
            dut,
            top="sc_lif_neuron",
            taps=[
                StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "refractory_counter", msb="31"),
            ],
        )
        ref = expose_state_taps(
            ref,
            top="sc_lif_reference",
            taps=[
                StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True),
                StateTap("refr_state", "32'd0", msb="31"),
            ],
        )
        common = {"DATA_WIDTH": 16, "FRACTION": 8, "V_REST": 0, "V_RESET": 0, "V_THRESHOLD": 256}
        ports = parse_module_interface(ref, "sc_lif_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_lif_neuron",
            ref_top="sc_lif_reference",
            dut_params={**common, "REFRACTORY_PERIOD": 0},
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_toolkit_generalises_to_quadratic_qif_unbounded(self, tmp_path: Path) -> None:
        """The whitebox-tap + multiplier-abstraction flow generalises to the QIF.

        A quadratic integrate-and-fire is a second neuron shape: its state update
        contains a ``v*v`` self-multiply (the LIF only multiplied state by a free
        input) declared inline as ``wire = expr``. Abstracting that product to a
        shared free input and tapping the single membrane state proves the
        structurally-distinct DUT and golden reference equivalent unbounded.
        """
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            _QIF_DUT,
            top="sc_qif_dut",
            signals=[LiftedSignal("v_squared", "v_sq_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        ref = abstract_to_free_inputs(
            _QIF_REF,
            top="sc_qif_reference",
            signals=[LiftedSignal("v_sq", "v_sq_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        dut = expose_state_taps(
            dut,
            top="sc_qif_dut",
            taps=[StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True)],
        )
        ref = expose_state_taps(
            ref,
            top="sc_qif_reference",
            taps=[StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True)],
        )
        common = {
            "DATA_WIDTH": 16,
            "K_SHIFT": 6,
            "V_THRESHOLD": 1024,
            "V_RESET": -1024,
            "V_MIN": -2048,
        }
        ports = parse_module_interface(ref, "sc_qif_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_qif_dut",
            ref_top="sc_qif_reference",
            dut_params=common,
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_toolkit_generalises_to_two_state_izhikevich_unbounded(self, tmp_path: Path) -> None:
        """The whitebox-tap + multiplier-abstraction flow generalises to two coupled states.

        Izhikevich is the first two-state neuron: membrane ``v`` and recovery ``u``
        evolve together, the quadratic ``(v-VR)*(v-VT)`` product drives ``v`` and a
        spike resets *both* (``v <- C``, ``u <- u + D``). Because ``u`` feeds
        ``v``'s next value, the unbounded proof needs the *coordinated* two-register
        state-matching invariant: tapping only ``v`` leaves the induction step free
        to start from a state where ``v`` agrees but ``u`` does not, and it diverges.
        Tapping *both* registers — plus abstracting the one quadratic product — proves
        the structurally-distinct DUT and golden reference equivalent unbounded.
        """
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            _IZH_DUT,
            top="sc_izh_dut",
            signals=[LiftedSignal("p_prod", "q_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        ref = abstract_to_free_inputs(
            _IZH_REF,
            top="sc_izh_reference",
            signals=[LiftedSignal("q_prod", "q_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        dut = expose_state_taps(
            dut,
            top="sc_izh_dut",
            taps=[
                StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True),
                StateTap("u_state", "u_reg", msb="DATA_WIDTH-1", signed=True),
            ],
        )
        ref = expose_state_taps(
            ref,
            top="sc_izh_reference",
            taps=[
                StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True),
                StateTap("u_state", "u", msb="DATA_WIDTH-1", signed=True),
            ],
        )
        common = {
            "DATA_WIDTH": 16,
            "KQ_SHIFT": 6,
            "KA_SHIFT": 5,
            "VR": -60,
            "VT": -40,
            "VPEAK": 30,
            "C_RESET": -50,
            "D_STEP": 6,
        }
        ports = parse_module_interface(ref, "sc_izh_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_izh_dut",
            ref_top="sc_izh_reference",
            dut_params=common,
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is True
        assert result.verdict == "PASS"
        assert result.mode == "prove"

    def test_two_state_needs_both_taps_to_converge(self, tmp_path: Path) -> None:
        """Tapping only ``v`` leaves the two-state induction incomplete.

        The companion to ``test_toolkit_generalises_to_two_state_izhikevich_unbounded``:
        it exposes *why* both coupled registers must be tapped. With the same
        abstraction but only the ``v`` tap, the reachable-state invariant omits
        ``u``; the induction step may start from a state where the membranes agree
        but the recovery variables differ, and since ``u`` feeds ``v``'s next value
        the k-induction cannot close. The verdict is ``UNKNOWN`` (inconclusive —
        base case holds, induction does not converge), not ``FAIL``: the modules are
        genuinely equivalent, the invariant is merely too weak to prove it. This is
        an inconclusive proof, not a tool failure, so it does not raise.
        """
        from sc_neurocore.compiler.equivalence_miter import parse_module_interface
        from sc_neurocore.compiler.operator_abstraction import (
            LiftedSignal,
            abstract_to_free_inputs,
        )
        from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

        dut = abstract_to_free_inputs(
            _IZH_DUT,
            top="sc_izh_dut",
            signals=[LiftedSignal("p_prod", "q_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        ref = abstract_to_free_inputs(
            _IZH_REF,
            top="sc_izh_reference",
            signals=[LiftedSignal("q_prod", "q_in", msb="2*DATA_WIDTH-1", signed=True)],
        )
        dut = expose_state_taps(
            dut,
            top="sc_izh_dut",
            taps=[StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True)],
        )
        ref = expose_state_taps(
            ref,
            top="sc_izh_reference",
            taps=[StateTap("v_state", "v", msb="DATA_WIDTH-1", signed=True)],
        )
        common = {
            "DATA_WIDTH": 16,
            "KQ_SHIFT": 6,
            "KA_SHIFT": 5,
            "VR": -60,
            "VT": -40,
            "VPEAK": 30,
            "C_RESET": -50,
            "D_STEP": 6,
        }
        ports = parse_module_interface(ref, "sc_izh_reference", params={"DATA_WIDTH": 16})
        result = prove_equivalence(
            dut,
            ref,
            ports,
            dut_top="sc_izh_dut",
            ref_top="sc_izh_reference",
            dut_params=common,
            ref_params=common,
            mode="prove",
            depth=6,
            workdir=tmp_path,
        )
        assert result.proven is False
        assert result.verdict == "UNKNOWN"

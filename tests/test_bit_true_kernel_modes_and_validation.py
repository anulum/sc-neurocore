# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModesAndValidation from former test_bit_true_kernel.py

"""Focused suite: TestModesAndValidation from former test_bit_true_kernel.py."""

from __future__ import annotations

from sc_neurocore.compiler.intelligence.bit_true_kernel import kernel_arithmetic_contract
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS
from tests.bit_true_kernel_support import *  # noqa: F403


def test_arithmetic_contract_refuses_an_unknown_function_lowering() -> None:
    """The public receipt validates its function lowering independently of model construction."""
    with pytest.raises(ValueError, match="unknown function_lowering"):
        kernel_arithmetic_contract(function_lowering="unknown")


def test_kernel_refuses_a_changed_unsupported_model_method() -> None:
    """A mutable public model cannot silently compile under an unsupported integration rule."""
    neuron = _lif()
    neuron.method = "unsupported"
    with pytest.raises(ValueError, match="currently supports method"):
        generate_bittrue_kernel_from_neuron(neuron, "m")


class TestModesAndValidation:
    """Rounding, overflow and configuration modes the bit-true kernel emits or refuses."""

    def test_nearest_rounding_uses_sign_aware_bias_in_c(self) -> None:
        """Nearest rounding biases a negative product by half minus one in C."""
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", rounding="nearest")
        assert "((int64_t)1 << (FRAC_BITS - 1))" in code
        assert "product < 0 ? half - 1 : half" in code

    def test_wrap_overflow_uses_sc_wrap(self) -> None:
        """The wrap overflow policy commits through sc_wrap."""
        code = generate_bittrue_kernel_from_neuron(_lif(), "sc_lif", overflow="wrap")
        assert "sc_wrap(((int64_t)s->v)" in code

    def test_nearest_rounding_uses_sign_aware_bias_in_rust(self) -> None:
        """Nearest rounding biases a negative product by half minus one in Rust."""
        code = generate_bittrue_kernel_from_neuron(
            _lif(), "sc_lif", rounding="nearest", language="rust"
        )
        assert "1i64 << (FRAC_BITS - 1)" in code
        assert "if product < 0 { half - 1 } else { half }" in code

    @pytest.mark.parametrize("language", ["c", "rust"])
    def test_nearest_rounding_with_zero_fraction_emits_no_negative_shift(
        self, language: str
    ) -> None:
        """With no fraction bits nearest rounding adds no bias and no negative shift."""
        code = generate_bittrue_kernel_from_neuron(
            _lif(),
            "sc_lif_integer",
            fraction=0,
            rounding="nearest",
            language=language,
        )

        assert "FRAC_BITS - 1" not in code
        assert "product < 0" not in code

    def test_bad_language_simple(self) -> None:
        """The derivative-mapping kernel refuses a language other than C or Rust."""
        with pytest.raises(ValueError, match="language must be"):
            generate_bittrue_kernel("m", {"v": "v"}, language="go")

    def test_bad_language_neuron(self) -> None:
        """The whole-neuron kernel refuses a language other than C or Rust."""
        with pytest.raises(ValueError, match="language must be"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", language="go")

    def test_bankers_rounding_rejected(self) -> None:
        """Bankers rounding, which only the RTL emits, is refused."""
        with pytest.raises(ValueError, match="rounding"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", rounding="bankers")

    def test_stochastic_rounding_rejected(self) -> None:
        """Stochastic product rounding is refused."""
        with pytest.raises(ValueError, match="stochastic"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", rounding="stochastic")

    def test_trap_overflow_rejected(self) -> None:
        """The trap overflow policy is refused."""
        with pytest.raises(ValueError, match="trap"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", overflow="trap")

    def test_unsigned_rejected(self) -> None:
        """Unsigned state is refused, as the RTL neuron contract is signed."""
        with pytest.raises(ValueError, match="signed=True"):
            generate_bittrue_kernel_from_neuron(_lif(), "m", signed=False)

    def test_dt_underflow_rejected(self) -> None:
        """A dt that encodes to zero in the format is refused."""
        # dt=0.001 underflows Q8.8 (resolution 1/256 ≈ 0.0039)
        neuron = from_equations("dv/dt = -v + I", init=dict(v=0.0), dt=0.001)
        with pytest.raises(ValueError, match="underflows"):
            generate_bittrue_kernel_from_neuron(neuron, "m", data_width=16, fraction=8)

    def test_every_model_integrator_is_mirrored(self) -> None:
        """Each integrator a neuron accepts has a kernel (RTL equality: test_bit_true_kernel_integrators)."""
        for method in SUPPORTED_METHODS:
            neuron = EquationNeuron(
                equations={"v": "-v + I"},
                state={"v": 0.0},
                dt=0.1,
                method=method,
            )
            assert "m_step" in generate_bittrue_kernel_from_neuron(neuron, "m")
        with pytest.raises(ValueError, match="supports method"):
            kernel_arithmetic_contract(method="leapfrog")

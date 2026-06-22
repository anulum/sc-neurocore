# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for live-control MMIO spec invariants

"""Contracts for TrapSpec, ParameterBankSpec and MMIOUpdateSpec validation invariants."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.compiler.live_control_specs import (
    MMIOUpdateSpec,
    ParameterBankSpec,
    TrapSpec,
)


def _bank(**overrides: Any) -> ParameterBankSpec:
    """A valid parameter bank seated above the control-register window."""
    base: dict[str, Any] = {
        "bank_name": "bank",
        "start_address_bytes": 64,
        "parameter_count": 2,
        "parameter_names": ("p0", "p1"),
    }
    base.update(overrides)
    return ParameterBankSpec(**base)


def _mmio(**overrides: Any) -> MMIOUpdateSpec:
    """A valid MMIO update spec with one non-overlapping bank."""
    base: dict[str, Any] = {"bus_protocol": "axi4_lite", "banks": (_bank(),)}
    base.update(overrides)
    return MMIOUpdateSpec(**base)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_flags": 0}, "max_flags must be a positive integer"),
        ({"max_flags": 300}, "max_flags cannot exceed 256"),
        ({"flag_register_offset": "nope"}, "flag_register_offset must be an integer"),
        ({"flag_register_offset": 6}, "4-byte aligned"),
    ],
)
def test_trap_spec_rejects_invalid_fields(overrides: dict[str, Any], match: str) -> None:
    """TrapSpec validates its flag count and register-offset alignment."""
    with pytest.raises(ValueError, match=match):
        TrapSpec(**overrides)


def test_parameter_bank_normalises_list_names_to_tuple() -> None:
    """A list of parameter names is frozen into a tuple by __post_init__."""
    bank = _bank(parameter_names=["p0", "p1"])

    assert bank.parameter_names == ("p0", "p1")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"parameter_names": ("p0", "")}, "non-empty strings"),
        ({"bank_name": "  "}, "bank_name must be a non-empty string"),
        ({"start_address_bytes": -4}, "non-negative integer"),
        ({"start_address_bytes": 2}, "4-byte aligned"),
        ({"parameter_count": 0}, "parameter_count must be a positive integer"),
        ({"parameter_names": ("p0", "p0")}, "parameter_names must be unique"),
        ({"parameter_names": (), "parameter_count": 1}, "at least one parameter"),
        ({"parameter_names": ("p0", "p1", "p2")}, "must not exceed parameter_count"),
        ({"precision_mode": "bogus"}, "precision_mode must be"),
        ({"q_format": "Q4.5"}, "Q-format width must be byte-aligned"),
        ({"precision_mode": "bfp", "bfp_exponent_bits": 2}, "bfp_exponent_bits must be between"),
        ({"precision_mode": "bfp", "bfp_mantissa_bits": 2}, "bfp_mantissa_bits must be between"),
        ({"precision_mode": "bfp", "bfp_exponent_bits": 3, "bfp_mantissa_bits": 4}, "byte-aligned"),
        ({"q_format": "Q40.32"}, "must not exceed 64 bits"),
        ({"writable": 1}, "writable must be a bool"),
    ],
)
def test_parameter_bank_rejects_invalid_fields(overrides: dict[str, Any], match: str) -> None:
    """ParameterBankSpec enforces naming, addressing, precision and width invariants."""
    with pytest.raises(ValueError, match=match):
        _bank(**overrides)


def test_parameter_bank_signed_code_bounds_and_word_normalisation() -> None:
    """The bank reports signed-code bounds and normalises signed words into storage words."""
    bank = _bank()

    assert bank.signed_code_min == -(1 << (bank.entry_width_bits - 1))
    assert bank.signed_code_max == (1 << (bank.entry_width_bits - 1)) - 1
    assert bank.normalise_encoded_word(-1) == bank.encoded_word_max


def test_parameter_bank_rejects_non_integer_encoded_word() -> None:
    """A boolean reset value is rejected as a non-integer encoded word."""
    with pytest.raises(ValueError, match="encoded parameter value must be an integer"):
        _bank(reset_value=True)


def test_parameter_bank_entry_index_guards() -> None:
    """entry_index rejects booleans, unknown names, foreign types and out-of-range indices."""
    bank = _bank()

    with pytest.raises(ValueError, match="must not be bool"):
        bank.entry_index(True)
    with pytest.raises(ValueError, match="unknown parameter"):
        bank.entry_index("absent")
    foreign: Any = 1.5
    with pytest.raises(ValueError, match="integer index or parameter name"):
        bank.entry_index(foreign)
    with pytest.raises(ValueError, match="parameter index out of range"):
        bank.entry_index(99)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"banks": ()}, "requires at least one ParameterBankSpec"),
        ({"banks": (object(),)}, "only ParameterBankSpec instances"),
        ({"read_data_width": 7}, "read_data_width must be one of"),
        ({"write_data_width": 7}, "write_data_width must be one of"),
        ({"address_width_bits": 8}, "address_width_bits must be between 12 and 64"),
        ({"bank_name_width": 4}, "bank_name_width must be between 8 and 64"),
        ({"supports_burst": 1}, "supports_burst must be a bool"),
        ({"supports_partial_write": 1}, "supports_partial_write must be a bool"),
        ({"trap": "nope"}, "trap must be a TrapSpec"),
        ({"control_base_address_bytes": 2}, "control_base_address_bytes must be"),
    ],
)
def test_mmio_update_spec_rejects_invalid_fields(overrides: dict[str, Any], match: str) -> None:
    """MMIOUpdateSpec validates banks, data widths, address widths and control placement."""
    with pytest.raises(ValueError, match=match):
        _mmio(**overrides)


def test_mmio_rejects_bank_name_width_below_longest_name() -> None:
    """The bank-name field must be wide enough for the longest registered bank name."""
    wide_name_bank = _bank(bank_name="an_extremely_long_bank_name")

    with pytest.raises(ValueError, match="bank_name_width too small"):
        _mmio(banks=(wide_name_bank,), bank_name_width=8)


def test_mmio_rejects_address_width_too_small_for_map() -> None:
    """A 12-bit address space cannot reach a bank seated far above its ceiling."""
    far_bank = _bank(start_address_bytes=8192)

    with pytest.raises(ValueError, match="address_width_bits too small"):
        _mmio(banks=(far_bank,), address_width_bits=12)

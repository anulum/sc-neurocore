# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The fixed-point LFSR refuses a register it could never leave

"""The bit-true LFSR refuses a zero register and a width below its lowest tap."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder, FixedPointLFSR


@pytest.mark.parametrize(("width", "seed"), [(16, 0x10000), (16, 0x30000), (8, 0x100), (6, 0xC0)])
def test_a_seed_whose_low_width_bits_are_zero_is_refused(width: int, seed: int) -> None:
    """A non-zero seed with no set bit in the register's ``width`` bits would load a stuck register.

    Parameters
    ----------
    width : int
        Register width.
    seed : int
        A non-zero seed whose low ``width`` bits are all 0.
    """
    with pytest.raises(ValueError, match="low 16 bits|low 8 bits|low 6 bits"):
        FixedPointLFSR(width=width, seed=seed)


def test_a_seed_wider_than_the_register_keeps_its_low_bits() -> None:
    """A seed with set bits inside the register is still masked to ``width`` bits, as the RTL parameter is."""
    lfsr = FixedPointLFSR(width=16, seed=0x1ACE1)
    assert lfsr.reg == 0xACE1
    assert lfsr.step() == FixedPointLFSR(width=16, seed=0xACE1).step()


@pytest.mark.parametrize("seed", [0, 0x10000])
def test_reset_refuses_a_zero_register_and_keeps_the_state(seed: int) -> None:
    """``reset`` checks the masked seed as the constructor does, and a refused reset changes nothing.

    Parameters
    ----------
    seed : int
        A seed whose low 16 bits are all 0.
    """
    lfsr = FixedPointLFSR(seed=0xACE1)
    lfsr.step()
    before = lfsr.reg
    with pytest.raises(ValueError, match="low 16 bits"):
        lfsr.reset(seed)
    assert lfsr.reg == before


def test_reset_without_a_seed_returns_to_the_constructor_seed() -> None:
    """``reset()`` reloads the constructor's seed, masked."""
    lfsr = FixedPointLFSR(seed=0x1ACE1)
    for _ in range(5):
        lfsr.step()
    lfsr.reset()
    assert lfsr.reg == 0xACE1


@pytest.mark.parametrize("width", [0, 1, 5])
def test_a_width_below_the_lowest_tap_is_refused_at_construction(width: int) -> None:
    """The feedback reads bit ``width - 6``, so a register narrower than 6 bits is refused when built.

    Parameters
    ----------
    width : int
        A register width below 6.
    """
    with pytest.raises(ValueError, match="at least 6"):
        FixedPointLFSR(width=width, seed=1)
    with pytest.raises(ValueError, match="at least 6"):
        FixedPointBitstreamEncoder(data_width=width, seed_init=1)


def test_the_narrowest_register_steps() -> None:
    """A 6-bit register has every tap and steps without error."""
    lfsr = FixedPointLFSR(width=6, seed=0b100001)
    states = {lfsr.step() for _ in range(70)}
    assert 0 not in states
    assert all(0 < state < 64 for state in states)

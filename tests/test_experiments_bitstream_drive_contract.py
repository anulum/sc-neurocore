# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for bitstream-drive experiment contract

"""Behavioural contract for the bitstream-driven LIF experiment."""

from __future__ import annotations


def test_bitstream_drive_returns_bounded_input_and_spike_probabilities() -> None:
    from sc_neurocore.experiments.bitstream_drive import run_bitstream_driven_lif

    input_bits, spike_bits, p_in, p_fire = run_bitstream_driven_lif(
        x_input=0.05,
        x_min=0.0,
        x_max=0.1,
        length=256,
    )

    assert input_bits.shape == (256,)
    assert spike_bits.shape == (256,)
    assert 0.0 <= p_in <= 1.0
    assert 0.0 <= p_fire <= 1.0

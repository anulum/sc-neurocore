# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_antagonistic) from former test_formal_network_properties.py

"""Tests of compiling antagonistic-output exclusion to SVA."""

from __future__ import annotations

from typing import Any
import pytest
from sc_neurocore.formal.network_properties import (
    DenseLIFNetworkSpec,
    NetworkAntagonisticOutputExclusion,
)
from sc_neurocore.formal.property_compiler import (
    compile_network_antagonistic_exclusion_sva,
)


def test_compile_dense_lif_antagonistic_exclusion_sva_is_deterministic() -> None:
    """The exclusion SVA is deterministic."""
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkAntagonisticOutputExclusion(
        name="motor_left_right_exclusion",
        output_a=0,
        output_b=1,
    )

    sva = compile_network_antagonistic_exclusion_sva(spec, prop)

    assert sva == compile_network_antagonistic_exclusion_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_antagonistic_sva" in sva
    assert "wire scnc_antagonist_a = spike_out[0];" in sva
    assert "wire scnc_antagonist_b = spike_out[1];" in sva
    assert (
        "a_motor_left_right_exclusion: assert (!(scnc_antagonist_a && scnc_antagonist_b));" in sva
    )
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_a": -1}, "output_a"),
        ({"output_b": -1}, "output_b"),
        ({"output_a": 1, "output_b": 1}, "distinct"),
    ],
)
def test_antagonistic_exclusion_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    """Each invalid exclusion field is rejected."""
    values: dict[str, Any] = {
        "name": "motor_left_right_exclusion",
        "output_a": 0,
        "output_b": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkAntagonisticOutputExclusion(**values)


def test_compiler_rejects_antagonistic_exclusion_outside_network_output_width() -> None:
    """An output past the network width is rejected."""
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkAntagonisticOutputExclusion(
        name="bad_exclusion",
        output_a=0,
        output_b=2,
    )

    with pytest.raises(ValueError, match="output_b"):
        compile_network_antagonistic_exclusion_sva(spec, prop)


def test_antagonistic_exclusion_rejects_output_index_beyond_width() -> None:
    """An output index past the width is rejected."""
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    exclusion = NetworkAntagonisticOutputExclusion(
        name="exclusion_out_of_range",
        output_a=5,
        output_b=1,
    )

    with pytest.raises(ValueError, match="output_a must refer to an existing"):
        compile_network_antagonistic_exclusion_sva(spec, exclusion)

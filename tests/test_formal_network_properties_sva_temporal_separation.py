# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_temporal_separation) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

def test_compile_dense_lif_temporal_separation_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    sva = compile_network_temporal_separation_sva(spec, prop)

    assert sva == compile_network_temporal_separation_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_temporal_separation_sva" in sva
    assert "parameter int unsigned SCNC_SEPARATION_CYCLES = 2;" in sva
    assert "wire scnc_temporal_a = spike_out[0];" in sva
    assert "wire scnc_temporal_b = spike_out[1];" in sva
    assert "wire scnc_after_a_active = scnc_after_a_count != '0;" in sva
    assert "wire scnc_after_b_active = scnc_after_b_count != '0;" in sva
    assert "a_motor_left_right_temporal_separation: assert" in sva
    assert "!(scnc_temporal_a && scnc_temporal_b)" in sva
    assert "!(scnc_temporal_a && scnc_after_b_active)" in sva
    assert "!(scnc_temporal_b && scnc_after_a_active)" in sva
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_a": -1}, "output_a"),
        ({"output_b": -1}, "output_b"),
        ({"output_a": 1, "output_b": 1}, "distinct"),
        ({"separation_cycles": 0}, "separation_cycles"),
    ],
)
def test_temporal_separation_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "motor_left_right_temporal_separation",
        "output_a": 0,
        "output_b": 1,
        "separation_cycles": 2,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkOutputTemporalSeparation(**values)


def test_compiler_rejects_temporal_separation_outside_network_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkOutputTemporalSeparation(
        name="bad_temporal_separation",
        output_a=0,
        output_b=2,
        separation_cycles=2,
    )

    with pytest.raises(ValueError, match="output_b"):
        compile_network_temporal_separation_sva(spec, prop)


def test_temporal_separation_rejects_output_index_beyond_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    separation = NetworkOutputTemporalSeparation(
        name="separation_out_of_range",
        output_a=5,
        output_b=1,
        separation_cycles=4,
    )

    with pytest.raises(ValueError, match="output_a must refer to an existing"):
        compile_network_temporal_separation_sva(spec, separation)



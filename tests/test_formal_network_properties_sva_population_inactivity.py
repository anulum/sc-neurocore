# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_population_inactivity) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

def test_compile_dense_lif_population_inactivity_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=3,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkPopulationInactivityBound(
        name="population_inactivity_bound",
        max_silent_cycles=2,
    )

    sva = compile_network_population_inactivity_sva(spec, prop)

    assert sva == compile_network_population_inactivity_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_population_inactivity_sva" in sva
    assert "parameter int unsigned SCNC_MAX_SILENT_CYCLES = 2;" in sva
    assert "assign scnc_active_outputs = spike_out[0] + spike_out[1] + spike_out[2];" in sva
    assert "wire scnc_no_active_outputs = scnc_active_outputs == '0;" in sva
    assert "assign scnc_next_silent_count" in sva
    assert "a_population_inactivity_bound: assert" in sva
    assert "scnc_next_silent_count <= SCNC_MAX_SILENT_CYCLES" in sva
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"max_silent_cycles": 0}, "max_silent_cycles"),
        ({"max_silent_cycles": True}, "max_silent_cycles"),
    ],
)
def test_population_inactivity_bound_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "population_inactivity_bound",
        "max_silent_cycles": 2,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkPopulationInactivityBound(**values)



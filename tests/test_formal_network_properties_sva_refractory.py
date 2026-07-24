# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_refractory) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403

def test_compile_dense_lif_refractory_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRefractoryInvariant(
        name="output1_refractory",
        output_index=1,
        refractory_cycles=3,
    )

    sva = compile_network_refractory_sva(spec, prop)

    assert sva == compile_network_refractory_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_refractory_sva" in sva
    assert "parameter int unsigned SCNC_REFRACTORY_CYCLES = 3;" in sva
    assert "wire scnc_monitored_spike = spike_out[1];" in sva
    assert "logic [$clog2(SCNC_REFRACTORY_CYCLES + 1)-1:0] scnc_refractory_count;" in sva
    assert "a_output1_refractory: assert (!scnc_monitored_spike);" in sva
    assert "if (rst_n && sample_valid && scnc_refractory_active) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_index": -1}, "output_index"),
        ({"refractory_cycles": 0}, "refractory_cycles"),
    ],
)
def test_refractory_invariant_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "output0_refractory",
        "output_index": 0,
        "refractory_cycles": 3,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkRefractoryInvariant(**values)


def test_compiler_rejects_refractory_outside_network_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkRefractoryInvariant(
        name="output2_refractory",
        output_index=2,
        refractory_cycles=3,
    )

    with pytest.raises(ValueError, match="output_index"):
        compile_network_refractory_sva(spec, prop)



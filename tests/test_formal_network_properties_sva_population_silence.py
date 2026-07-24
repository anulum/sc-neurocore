# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_population_silence) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403


def test_compile_dense_lif_population_silence_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=3,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=2,
        silence_cycles=3,
    )

    sva = compile_network_population_silence_sva(spec, prop)

    assert sva == compile_network_population_silence_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_population_silence_sva" in sva
    assert "parameter int unsigned SCNC_TRIGGER_ACTIVE_OUTPUTS = 2;" in sva
    assert "parameter int unsigned SCNC_SILENCE_CYCLES = 3;" in sva
    assert "assign scnc_active_outputs = spike_out[0] + spike_out[1] + spike_out[2];" in sva
    assert "wire scnc_coactivation_trigger" in sva
    assert "wire scnc_silence_active = scnc_silence_count != '0;" in sva
    assert "a_population_silence_after_coactivation: assert (scnc_active_outputs == '0);" in sva
    assert "if (rst_n && sample_valid && scnc_silence_active) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"trigger_active_outputs": 0}, "trigger_active_outputs"),
        ({"trigger_active_outputs": True}, "trigger_active_outputs"),
        ({"silence_cycles": 0}, "silence_cycles"),
    ],
)
def test_population_silence_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "population_silence_after_coactivation",
        "trigger_active_outputs": 2,
        "silence_cycles": 3,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkPopulationSilenceAfterCoactivation(**values)


def test_compiler_rejects_population_silence_trigger_above_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=3,
        silence_cycles=2,
    )

    with pytest.raises(ValueError, match="trigger_active_outputs"):
        compile_network_population_silence_sva(spec, prop)

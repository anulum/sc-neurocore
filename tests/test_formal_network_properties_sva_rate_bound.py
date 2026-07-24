# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sva_rate_bound) from former test_formal_network_properties.py

from __future__ import annotations

from tests.formal_network_properties_support import *  # noqa: F403


def test_compile_dense_lif_rate_bound_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=16,
        max_spikes=3,
    )

    sva = compile_network_rate_bound_sva(spec, prop)

    assert sva == compile_network_rate_bound_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_rate_bound_sva" in sva
    assert "parameter int unsigned SCNC_WINDOW_CYCLES = 16;" in sva
    assert "parameter int unsigned SCNC_MAX_SPIKES = 3;" in sva
    assert "logic [$clog2(SCNC_WINDOW_CYCLES + 1)-1:0] scnc_window_count;" in sva
    assert "logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_spike_count;" in sva
    assert "logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_next_spike_count;" in sva
    assert "wire scnc_monitored_spike = spike_out[0];" in sva
    assert "assign scnc_next_spike_count = scnc_spike_count + scnc_monitored_spike;" in sva
    assert "a_output0_rate_bound: assert (scnc_next_spike_count <= SCNC_MAX_SPIKES);" in sva
    assert "SCNC_WINDOW_CYCLES - 1" in sva
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


def test_compile_dense_lif_fixture_rtl_is_deterministic_and_matches_formal_ports() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )

    rtl = compile_dense_lif_fixture_rtl(spec)

    assert rtl == compile_dense_lif_fixture_rtl(spec)
    assert "module dense_lif_frontier_fixture (" in rtl
    assert "input logic clk," in rtl
    assert "input logic rst_n," in rtl
    assert "input logic sample_valid," in rtl
    assert "input logic [2:0] spike_in," in rtl
    assert "output logic [1:0] spike_out" in rtl
    assert "logic signed [15:0] membrane [0:1];" in rtl
    assert "localparam logic signed [15:0] SCNC_THRESHOLD = 16'sd256;" in rtl
    assert "dense_drive[0] = spike_in[0] + spike_in[1] + spike_in[2];" in rtl
    assert "membrane[0] <= '0;" in rtl
    assert "spike_out[0] <= 1'b1;" in rtl
    assert "spike_out[1] <= 1'b1;" in rtl


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"input_width": 0}, "input_width"),
        ({"output_width": 0}, "output_width"),
        ({"state_width": 0}, "state_width"),
        ({"timestep_name": "1tick"}, "timestep_name"),
        ({"output_signal": "spike-out"}, "output_signal"),
    ],
)
def test_dense_lif_spec_rejects_invalid_contracts(kwargs: dict[str, object], match: str) -> None:
    values: dict[str, object] = {
        "name": "dense_lif_frontier_fixture",
        "input_width": 3,
        "output_width": 2,
        "state_width": 16,
        "timestep_name": "sample_valid",
        "output_signal": "spike_out",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DenseLIFNetworkSpec(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"output_index": -1}, "output_index"),
        ({"window_cycles": 0}, "window_cycles"),
        ({"max_spikes": -1}, "max_spikes"),
        ({"window_cycles": 4, "max_spikes": 5}, "max_spikes"),
    ],
)
def test_rate_bound_rejects_invalid_contracts(kwargs: dict[str, object], match: str) -> None:
    values: dict[str, object] = {
        "name": "output0_rate_bound",
        "output_index": 0,
        "window_cycles": 16,
        "max_spikes": 3,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkRateBound(**values)


def test_compiler_rejects_rate_bound_outside_network_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkRateBound(
        name="output2_rate_bound",
        output_index=2,
        window_cycles=16,
        max_spikes=3,
    )

    with pytest.raises(ValueError, match="output_index"):
        compile_network_rate_bound_sva(spec, prop)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for network-level formal property compilation

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.formal.network_properties import (
    DenseLIFNetworkSpec,
    NetworkAntagonisticOutputExclusion,
    NetworkOutputTemporalSeparation,
    NetworkPopulationCoactivationCap,
    NetworkPopulationInactivityBound,
    NetworkPopulationSilenceAfterCoactivation,
    NetworkRefractoryInvariant,
    NetworkRateBound,
)
from sc_neurocore.formal.property_compiler import (
    compile_dense_lif_fixture_rtl,
    compile_network_antagonistic_exclusion_sva,
    compile_network_population_coactivation_sva,
    compile_network_population_inactivity_sva,
    compile_network_population_silence_sva,
    compile_network_temporal_separation_sva,
    compile_network_rate_bound_sva,
    compile_network_refractory_sva,
)
from sc_neurocore.formal.counterexample_replay import (
    replay_antagonistic_counterexample,
    replay_population_coactivation_counterexample,
    replay_population_inactivity_counterexample,
    replay_population_silence_counterexample,
    replay_temporal_separation_counterexample,
    replay_rate_bound_counterexample,
    replay_refractory_counterexample,
)
from sc_neurocore.formal.report_schema import (
    FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
    FormalReportValidationError,
    validate_formal_network_report,
)


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
def test_dense_lif_spec_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
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
def test_rate_bound_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
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


def test_compile_dense_lif_antagonistic_exclusion_sva_is_deterministic() -> None:
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
    assert "a_motor_left_right_exclusion: assert (!(scnc_antagonist_a && scnc_antagonist_b));" in sva
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
    values: dict[str, object] = {
        "name": "motor_left_right_exclusion",
        "output_a": 0,
        "output_b": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkAntagonisticOutputExclusion(**values)


def test_compiler_rejects_antagonistic_exclusion_outside_network_output_width() -> None:
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


def test_compile_dense_lif_population_coactivation_sva_is_deterministic() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=3,
        state_width=16,
        timestep_name="sample_valid",
        output_signal="spike_out",
    )
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    sva = compile_network_population_coactivation_sva(spec, prop)

    assert sva == compile_network_population_coactivation_sva(spec, prop)
    assert "module dense_lif_frontier_fixture_population_coactivation_sva" in sva
    assert "parameter int unsigned SCNC_MAX_ACTIVE_OUTPUTS = 1;" in sva
    assert "logic [1:0] scnc_active_outputs;" in sva
    assert "assign scnc_active_outputs = spike_out[0] + spike_out[1] + spike_out[2];" in sva
    assert "a_population_coactivation_cap: assert" in sva
    assert "scnc_active_outputs <= SCNC_MAX_ACTIVE_OUTPUTS" in sva
    assert "if (rst_n && sample_valid) begin" in sva
    assert "\nbind dense_lif_frontier_fixture" in sva


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad-name"}, "valid SystemVerilog identifier"),
        ({"max_active_outputs": -1}, "max_active_outputs"),
        ({"max_active_outputs": True}, "max_active_outputs"),
    ],
)
def test_population_coactivation_cap_rejects_invalid_contracts(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "name": "population_coactivation_cap",
        "max_active_outputs": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        NetworkPopulationCoactivationCap(**values)


def test_compiler_rejects_population_coactivation_cap_above_output_width() -> None:
    spec = DenseLIFNetworkSpec(
        name="dense_lif_frontier_fixture",
        input_width=3,
        output_width=2,
        state_width=16,
    )
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=3,
    )

    with pytest.raises(ValueError, match="max_active_outputs"):
        compile_network_population_coactivation_sva(spec, prop)


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


def test_counterexample_replay_detects_aligned_window_rate_violation() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, False, True, True], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.window_start_cycle == 0
    assert replay.observed_spikes == 3


def test_counterexample_replay_selects_monitored_output_index() -> None:
    prop = NetworkRateBound(
        name="output1_rate_bound",
        output_index=1,
        window_cycles=4,
        max_spikes=1,
    )

    replay = replay_rate_bound_counterexample(
        [
            [True, False],
            [True, True],
            [False, False],
            [False, True],
        ],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 3
    assert replay.observed_spikes == 2


def test_counterexample_replay_resets_on_aligned_windows() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    replay = replay_rate_bound_counterexample([True, True, False, False, True, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_spikes == 2


def test_counterexample_replay_rejects_non_binary_trace_values() -> None:
    prop = NetworkRateBound(
        name="output0_rate_bound",
        output_index=0,
        window_cycles=4,
        max_spikes=2,
    )

    with pytest.raises(ValueError, match="binary spike"):
        replay_rate_bound_counterexample([0, 2], prop)


def test_refractory_replay_detects_spike_inside_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=3,
    )

    replay = replay_refractory_counterexample([True, False, True, False], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0
    assert replay.remaining_refractory_cycles == 2
    assert replay.cycles_checked == 3


def test_refractory_replay_accepts_spike_after_refractory_window() -> None:
    prop = NetworkRefractoryInvariant(
        name="output0_refractory",
        output_index=0,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample([True, False, False, True], prop)

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.trigger_cycle is None
    assert replay.cycles_checked == 4


def test_refractory_replay_selects_monitored_output_index() -> None:
    prop = NetworkRefractoryInvariant(
        name="output1_refractory",
        output_index=1,
        refractory_cycles=2,
    )

    replay = replay_refractory_counterexample(
        [[True, True], [True, False], [False, True]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0


def test_antagonistic_replay_detects_simultaneous_outputs() -> None:
    prop = NetworkAntagonisticOutputExclusion(
        name="motor_left_right_exclusion",
        output_a=0,
        output_b=1,
    )

    replay = replay_antagonistic_counterexample(
        [[True, False], [False, True], [True, True]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.output_a == 0
    assert replay.output_b == 1
    assert replay.cycles_checked == 3


def test_antagonistic_replay_accepts_mutually_exclusive_outputs() -> None:
    prop = NetworkAntagonisticOutputExclusion(
        name="motor_left_right_exclusion",
        output_a=0,
        output_b=1,
    )

    replay = replay_antagonistic_counterexample(
        [[True, False], [False, True], [False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.cycles_checked == 3


def test_temporal_separation_replay_detects_bounded_window_violation() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[True, False], [False, True], [False, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 1
    assert replay.trigger_output == 0
    assert replay.violating_output == 1
    assert replay.remaining_separation_cycles == 2
    assert replay.cycles_checked == 2


def test_temporal_separation_replay_rejects_simultaneous_outputs() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample([[True, True]], prop)

    assert replay.violated
    assert replay.first_violation_cycle == 0
    assert replay.trigger_output is None
    assert replay.violating_output is None


def test_temporal_separation_replay_accepts_outputs_after_window() -> None:
    prop = NetworkOutputTemporalSeparation(
        name="motor_left_right_temporal_separation",
        output_a=0,
        output_b=1,
        separation_cycles=2,
    )

    replay = replay_temporal_separation_counterexample(
        [[True, False], [False, False], [False, False], [False, True]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.cycles_checked == 4


def test_population_coactivation_replay_detects_too_many_simultaneous_outputs() -> None:
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    replay = replay_population_coactivation_counterexample(
        [[True, False, True], [False, True, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 0
    assert replay.observed_active_outputs == 2
    assert replay.max_active_outputs == 1
    assert replay.cycles_checked == 1


def test_population_coactivation_replay_accepts_outputs_within_cap() -> None:
    prop = NetworkPopulationCoactivationCap(
        name="population_coactivation_cap",
        max_active_outputs=1,
    )

    replay = replay_population_coactivation_counterexample(
        [[True, False, False], [False, True, False], [False, False, True]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_active_outputs == 1
    assert replay.max_active_outputs == 1
    assert replay.cycles_checked == 3


def test_population_silence_replay_detects_spike_after_coactivation() -> None:
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=2,
        silence_cycles=2,
    )

    replay = replay_population_silence_counterexample(
        [[True, True, False], [False, False, False], [False, True, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 2
    assert replay.trigger_cycle == 0
    assert replay.observed_active_outputs == 1
    assert replay.remaining_silence_cycles == 1
    assert replay.trigger_active_outputs == 2
    assert replay.silence_cycles == 2
    assert replay.cycles_checked == 3


def test_population_silence_replay_accepts_silent_window() -> None:
    prop = NetworkPopulationSilenceAfterCoactivation(
        name="population_silence_after_coactivation",
        trigger_active_outputs=2,
        silence_cycles=2,
    )

    replay = replay_population_silence_counterexample(
        [[True, True, False], [False, False, False], [False, False, False], [True, False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.trigger_cycle is None
    assert replay.observed_active_outputs == 0
    assert replay.remaining_silence_cycles == 0
    assert replay.cycles_checked == 4


def test_population_inactivity_replay_detects_too_many_silent_cycles() -> None:
    prop = NetworkPopulationInactivityBound(
        name="population_inactivity_bound",
        max_silent_cycles=2,
    )

    replay = replay_population_inactivity_counterexample(
        [[False, False], [False, False], [True, False], [False, False], [False, False], [False, False]],
        prop,
    )

    assert replay.violated
    assert replay.first_violation_cycle == 5
    assert replay.observed_silent_cycles == 3
    assert replay.max_silent_cycles == 2
    assert replay.cycles_checked == 6


def test_population_inactivity_replay_accepts_bounded_silent_runs() -> None:
    prop = NetworkPopulationInactivityBound(
        name="population_inactivity_bound",
        max_silent_cycles=2,
    )

    replay = replay_population_inactivity_counterexample(
        [[False, False], [True, False], [False, False], [False, False]],
        prop,
    )

    assert not replay.violated
    assert replay.first_violation_cycle is None
    assert replay.observed_silent_cycles == 2
    assert replay.max_silent_cycles == 2
    assert replay.cycles_checked == 4


def _valid_formal_report_payload() -> dict[str, object]:
    return {
        "schema_version": FORMAL_NETWORK_REPORT_SCHEMA_VERSION,
        "network": {
            "name": "dense_lif_frontier_fixture",
            "input_width": 3,
            "output_width": 2,
            "state_width": 16,
            "timestep_name": "sample_valid",
            "output_signal": "spike_out",
            "clock_name": "clk",
            "reset_name": "rst_n",
        },
        "rate_bound": {
            "name": "output0_rate_bound",
            "output_index": 0,
            "window_cycles": 8,
            "max_spikes": 4,
        },
        "refractory": {
            "name": "output0_refractory",
            "output_index": 0,
            "refractory_cycles": 2,
        },
        "antagonistic_exclusion": {
            "name": "motor_left_right_exclusion",
            "output_a": 0,
            "output_b": 1,
        },
        "temporal_separation": {
            "name": "motor_left_right_temporal_separation",
            "output_a": 0,
            "output_b": 1,
            "separation_cycles": 2,
        },
        "population_coactivation": {
            "name": "population_coactivation_cap",
            "max_active_outputs": 1,
        },
        "population_silence": {
            "name": "population_silence_after_coactivation",
            "trigger_active_outputs": 2,
            "silence_cycles": 2,
        },
        "population_inactivity": {
            "name": "population_inactivity_bound",
            "max_silent_cycles": 2,
        },
        "artifacts": {
            "rtl": "/tmp/dense_lif_frontier_fixture.v",
            "sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "rate_sva": "/tmp/dense_lif_frontier_fixture_rate_bound.sv",
            "refractory_sva": "/tmp/dense_lif_frontier_fixture_refractory.sv",
            "antagonistic_sva": "/tmp/dense_lif_frontier_fixture_antagonistic.sv",
            "temporal_sva": "/tmp/dense_lif_frontier_fixture_temporal_separation.sv",
            "population_sva": "/tmp/dense_lif_frontier_fixture_population_coactivation.sv",
            "population_silence_sva": "/tmp/dense_lif_frontier_fixture_population_silence.sv",
            "population_inactivity_sva": "/tmp/dense_lif_frontier_fixture_population_inactivity.sv",
            "formal_bundle": "/tmp/dense_lif_frontier_fixture_formal_bundle.sv",
            "sby": "/tmp/dense_lif_frontier_fixture.sby",
            "report": "/tmp/formal_rate_bound_report.json",
        },
        "replay": {
            "violated": False,
            "first_violation_cycle": None,
            "window_start_cycle": None,
            "observed_spikes": 2,
            "cycles_checked": 4,
        },
        "rate_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "window_start_cycle": None,
            "observed_spikes": 2,
            "cycles_checked": 4,
        },
        "refractory_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_cycle": None,
            "remaining_refractory_cycles": 0,
            "cycles_checked": 4,
        },
        "antagonistic_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "output_a": 0,
            "output_b": 1,
            "cycles_checked": 4,
        },
        "temporal_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_output": None,
            "violating_output": None,
            "remaining_separation_cycles": 0,
            "cycles_checked": 4,
        },
        "population_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "observed_active_outputs": 1,
            "max_active_outputs": 1,
            "cycles_checked": 4,
        },
        "population_silence_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "trigger_cycle": None,
            "observed_active_outputs": 0,
            "remaining_silence_cycles": 0,
            "trigger_active_outputs": 2,
            "silence_cycles": 2,
            "cycles_checked": 4,
        },
        "population_inactivity_replay": {
            "violated": False,
            "first_violation_cycle": None,
            "observed_silent_cycles": 2,
            "max_silent_cycles": 2,
            "cycles_checked": 4,
        },
        "symbiyosys": {
            "requested": True,
            "status": "tool_unavailable",
            "command": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "sby": "/tmp/dense_lif_frontier_fixture.sby",
        },
    }


def test_validate_formal_network_report_accepts_complete_payload() -> None:
    payload = _valid_formal_report_payload()

    validate_formal_network_report(payload)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "schema_version"),
        (lambda payload: payload["artifacts"].pop("rtl"), "artifacts.rtl"),
        (lambda payload: payload["network"].__setitem__("output_width", 0), "output_width"),
        (lambda payload: payload["rate_bound"].__setitem__("max_spikes", 9), "max_spikes"),
        (lambda payload: payload["symbiyosys"].__setitem__("status", "unknown"), "symbiyosys.status"),
        (lambda payload: payload.__setitem__("rate_replay", {"violated": False}), "rate_replay"),
        (
            lambda payload: payload["temporal_replay"].__setitem__("trigger_output", 9),
            "temporal_replay.trigger_output",
        ),
        (
            lambda payload: payload["temporal_replay"].__setitem__("violating_output", 9),
            "temporal_replay.violating_output",
        ),
        (
            lambda payload: (
                payload["temporal_replay"].__setitem__("trigger_output", 0),
                payload["temporal_replay"].__setitem__("violating_output", 0),
            ),
            "temporal_replay.violating_output",
        ),
        (
            lambda payload: payload["population_coactivation"].__setitem__(
                "max_active_outputs", 3
            ),
            "population_coactivation.max_active_outputs",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__(
                "max_active_outputs", 2
            ),
            "population_replay.max_active_outputs",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__(
                "observed_active_outputs", 3
            ),
            "population_replay.observed_active_outputs",
        ),
        (
            lambda payload: (
                payload["population_replay"].__setitem__("violated", True),
                payload["population_replay"].__setitem__("first_violation_cycle", None),
                payload["population_replay"].__setitem__("observed_active_outputs", 2),
            ),
            "population_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_replay"].__setitem__(
                "observed_active_outputs", 2
            ),
            "population_replay.observed_active_outputs",
        ),
        (
            lambda payload: payload["population_silence"].__setitem__(
                "trigger_active_outputs", 3
            ),
            "population_silence.trigger_active_outputs",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "trigger_active_outputs", 3
            ),
            "population_silence_replay.trigger_active_outputs",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", None
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", None),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", 2
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
            ),
            "population_silence_replay.trigger_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 2),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", 2
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
            ),
            "population_silence_replay.trigger_cycle",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", 4
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "remaining_silence_cycles", 3
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", 2
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
                payload["population_silence_replay"].__setitem__(
                    "remaining_silence_cycles", 2
                ),
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__("cycles_checked", 2),
                payload["population_silence_replay"].__setitem__(
                    "remaining_silence_cycles", 2
                ),
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "remaining_silence_cycles", 1
            ),
            "population_silence_replay.remaining_silence_cycles",
        ),
        (
            lambda payload: (
                payload["population_silence_replay"].__setitem__("violated", True),
                payload["population_silence_replay"].__setitem__("trigger_cycle", 0),
                payload["population_silence_replay"].__setitem__(
                    "first_violation_cycle", 3
                ),
                payload["population_silence_replay"].__setitem__(
                    "observed_active_outputs", 1
                ),
                payload["population_silence_replay"].__setitem__("cycles_checked", 5),
            ),
            "population_silence_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_silence_replay"].__setitem__(
                "observed_active_outputs", 1
            ),
            "population_silence_replay.observed_active_outputs",
        ),
        (
            lambda payload: payload["population_inactivity"].__setitem__(
                "max_silent_cycles", 0
            ),
            "population_inactivity.max_silent_cycles",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "max_silent_cycles", 3
            ),
            "population_inactivity_replay.max_silent_cycles",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__(
                    "first_violation_cycle", None
                ),
                payload["population_inactivity_replay"].__setitem__(
                    "observed_silent_cycles", 3
                ),
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__(
                    "first_violation_cycle", 4
                ),
                payload["population_inactivity_replay"].__setitem__(
                    "observed_silent_cycles", 3
                ),
                payload["population_inactivity_replay"].__setitem__("cycles_checked", 4),
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: (
                payload["population_inactivity_replay"].__setitem__("violated", True),
                payload["population_inactivity_replay"].__setitem__(
                    "first_violation_cycle", 3
                ),
                payload["population_inactivity_replay"].__setitem__(
                    "observed_silent_cycles", 4
                ),
            ),
            "population_inactivity_replay.observed_silent_cycles",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "first_violation_cycle", 3
            ),
            "population_inactivity_replay.first_violation_cycle",
        ),
        (
            lambda payload: payload["population_inactivity_replay"].__setitem__(
                "observed_silent_cycles", 3
            ),
            "population_inactivity_replay.observed_silent_cycles",
        ),
    ],
)
def test_validate_formal_network_report_rejects_invalid_payloads(
    mutator, match: str
) -> None:
    payload = _valid_formal_report_payload()
    mutator(payload)

    with pytest.raises(FormalReportValidationError, match=match):
        validate_formal_network_report(payload)


def test_formal_network_verification_docs_cover_cli_and_report_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = repo_root / "docs" / "api" / "formal_network_verification.md"

    text = doc.read_text(encoding="utf-8")

    assert "sc-neurocore formal verify-network" in text
    assert "--refractory-cycles" in text
    assert "--run-symbiyosys" in text
    assert "formal_rate_bound_report.json" in text
    assert "FORMAL_NETWORK_REPORT_SCHEMA_VERSION" in text
    assert "validate_formal_network_report" in text
    assert "tools/verify_formal_network_evidence.py" in text
    assert "formal_network_coverage_manifest.json" in text
    assert "covered_outputs" in text
    assert "artifacts.rtl" in text
    assert "rate_replay" in text
    assert "refractory_replay" in text
    assert "antagonistic_exclusion" in text
    assert "temporal_separation" in text
    assert "population_coactivation" in text
    assert "population_silence" in text

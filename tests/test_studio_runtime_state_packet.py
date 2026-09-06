# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Foreign runtime state transport contract

"""What the Rust boundary may name, and what it must refuse.

Measured over the catalogue before this contract existed: the Rust batch lane
runs 158 of 185 models; of the 152 with a declared layout it carries 111
declared variables and drops 356; and 41 of them declare no variable called
``v`` at all, yet the payload placed the lane's ``soma_voltage()`` trace under
``states["v"]`` — while the layout beside it reported ``recorded: []``. The
payload and its own custody verdict disagreed.

The boundary also trusted what came back: a 100-step request against a
ten-sample drive returned ten voltages into a payload built for 100, and spike
indices were used as sample positions without a domain or ordering check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.studio.runtime_state_packet import (
    RUNTIME_STATE_PACKET_SCHEMA_VERSION,
    RUST_BATCH_PACKET,
    ForeignRuntimeError,
    RuntimeStatePacket,
    packet_coverage,
    validate_scalar_trace,
    validate_spike_indices,
)

#: A model whose declared state does not contain ``v`` at all.
NO_V_MODEL = "PinskyRinzelNeuron"

#: A model whose declared state contains ``v`` and more besides.
PARTIAL_MODEL = "AdExNeuron"


class TestThePacket:
    def test_the_rust_lane_states_what_it_carries(self) -> None:
        assert RUST_BATCH_PACKET.runtime == "rust-batch"
        assert RUST_BATCH_PACKET.exports == ("v",)
        assert RUST_BATCH_PACKET.carries_initial_snapshot is False
        assert RUST_BATCH_PACKET.carries_parameters is False

    def test_the_public_form_carries_its_contract_version(self) -> None:
        document = RUST_BATCH_PACKET.to_public_dict()

        assert document["schema_version"] == RUNTIME_STATE_PACKET_SCHEMA_VERSION
        assert document["exports"] == ["v"]
        assert document["runtime"] == "rust-batch"


class TestCoverage:
    def test_a_lane_that_carries_part_of_the_state_says_which_part(self) -> None:
        coverage = packet_coverage(RUST_BATCH_PACKET, ("v", "w"))

        assert coverage.carried == ("v",)
        assert coverage.dropped == ("w",)
        assert coverage.unnameable == ()
        assert coverage.complete is False
        assert coverage.names_nothing is False

    def test_a_lane_that_carries_all_of_it_is_complete(self) -> None:
        coverage = packet_coverage(RUST_BATCH_PACKET, ("v",))

        assert coverage.complete is True
        assert coverage.dropped == ()

    def test_an_export_the_model_does_not_declare_is_unnameable(self) -> None:
        """The 41-model case: the lane returns a value this model has no name for."""
        coverage = packet_coverage(RUST_BATCH_PACKET, ("v_s", "v_d", "h"))

        assert coverage.carried == ()
        assert coverage.unnameable == ("v",)
        assert coverage.names_nothing is True
        assert coverage.complete is False

    def test_declaration_order_is_preserved(self) -> None:
        packet = RuntimeStatePacket(
            runtime="test",
            exports=("b", "a"),
            carries_initial_snapshot=True,
            carries_parameters=True,
        )

        assert packet_coverage(packet, ("a", "b", "c")).carried == ("a", "b")

    def test_the_public_form_is_evidence_shaped(self) -> None:
        document = packet_coverage(RUST_BATCH_PACKET, ("v", "w")).to_public_dict()

        assert document == {
            "carried": ["v"],
            "complete": False,
            "dropped": ["w"],
            "runtime": "rust-batch",
            "schema_version": RUNTIME_STATE_PACKET_SCHEMA_VERSION,
            "unnameable": [],
        }


class TestValidatingATrace:
    def test_a_well_formed_trace_is_returned_as_float64(self) -> None:
        trace = validate_scalar_trace([1, 2, 3], runtime="rust-batch", name="v", n_steps=3)

        assert trace.dtype == np.float64
        assert list(trace) == [1.0, 2.0, 3.0]

    def test_a_short_trace_is_refused(self) -> None:
        """The measured case: ten samples returned into a 100-step payload."""
        with pytest.raises(ForeignRuntimeError, match="returned 10 samples of 'v' for a 100-step"):
            validate_scalar_trace(np.zeros(10), runtime="rust-batch", name="v", n_steps=100)

    def test_a_trace_that_is_not_a_single_series_is_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="not a single series"):
            validate_scalar_trace(np.zeros((2, 3)), runtime="rust-batch", name="v", n_steps=6)

    def test_a_trace_that_is_not_numeric_is_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="not numeric"):
            validate_scalar_trace(["a", "b"], runtime="rust-batch", name="v", n_steps=2)

    def test_a_non_finite_value_is_not_this_check_s_business(self) -> None:
        """A diverged voltage is a failed simulation, not a malformed boundary."""
        trace = validate_scalar_trace(
            [1.0, float("inf")], runtime="rust-batch", name="v", n_steps=2
        )

        assert np.isinf(trace[1])


class TestValidatingSpikeIndices:
    def test_well_formed_indices_are_returned_in_order(self) -> None:
        assert validate_spike_indices(
            np.array([0, 5, 9], dtype=np.uint64), runtime="rust-batch", n_steps=10
        ) == [0, 5, 9]

    def test_no_spikes_is_not_a_failure(self) -> None:
        assert validate_spike_indices(np.zeros(0), runtime="rust-batch", n_steps=10) == []

    def test_an_index_outside_the_run_is_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="outside a 10-step run"):
            validate_spike_indices(np.array([10]), runtime="rust-batch", n_steps=10)

    def test_a_negative_index_is_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="outside a 10-step run"):
            validate_spike_indices(np.array([-1]), runtime="rust-batch", n_steps=10)

    def test_indices_that_do_not_increase_are_refused(self) -> None:
        """The statistics take differences; an unordered series reports a negative interval."""
        with pytest.raises(ForeignRuntimeError, match="do not increase"):
            validate_spike_indices(np.array([5, 2]), runtime="rust-batch", n_steps=10)

    def test_a_repeated_index_is_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="do not increase"):
            validate_spike_indices(np.array([2, 2]), runtime="rust-batch", n_steps=10)

    def test_indices_that_are_not_positions_are_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="not positions"):
            validate_spike_indices(np.array([1.5]), runtime="rust-batch", n_steps=10)

    def test_indices_that_are_not_a_single_series_are_refused(self) -> None:
        with pytest.raises(ForeignRuntimeError, match="not a single series"):
            validate_spike_indices(
                np.zeros((2, 2), dtype=np.int64), runtime="rust-batch", n_steps=4
            )

    def test_something_that_is_not_an_array_is_refused(self) -> None:
        class Awkward:
            def __array__(self, dtype: Any = None) -> Any:
                raise ValueError("not an array")

        with pytest.raises(ForeignRuntimeError, match="not an array"):
            validate_spike_indices(Awkward(), runtime="rust-batch", n_steps=4)


class TestThroughTheRealBoundary:
    def test_a_model_the_lane_cannot_name_records_no_state_and_says_why(self) -> None:
        """The payload must not contradict the layout printed beside it."""
        simulate_model = pytest.importorskip("sc_neurocore.studio.model_simulate").simulate_model
        pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")

        result = simulate_model(NO_V_MODEL, duration=20.0, current=10.0)

        if result["effective_inputs"]["backend"] != "rust":
            pytest.skip("the Rust batch lane did not take this model")
        assert result["states"] == {}
        assert result["final_state"] == {}
        assert result["state_layout"]["recorded"] == []
        assert any("does not declare" in note for note in result["state_layout"]["custody_notes"])

    def test_a_model_the_lane_can_name_still_records_that_variable(self) -> None:
        simulate_model = pytest.importorskip("sc_neurocore.studio.model_simulate").simulate_model
        pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")

        result = simulate_model(PARTIAL_MODEL, duration=20.0, current=50.0)

        if result["effective_inputs"]["backend"] != "rust":
            pytest.skip("the Rust batch lane did not take this model")
        assert sorted(result["states"]) == ["v"]
        assert result["state_layout"]["recorded"] == ["v"]
        assert sorted(result["final_state"]) == ["v"]

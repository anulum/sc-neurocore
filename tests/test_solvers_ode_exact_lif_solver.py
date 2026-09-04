# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExactLIFSolver from former test_solvers_ode.py

"""Focused suite: TestExactLIFSolver from former test_solvers_ode.py."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import sc_neurocore.solvers.exact_lif_profile as lif_profile_module
from tests.solvers_ode_support import *  # noqa: F403

_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_RECEIPT = (
    _ROOT / "src/sc_neurocore/neurons/reference_trace_data/exact_current_lif_profile_v1.json"
)
_PACKET_RECEIPT = (
    _ROOT / "src/sc_neurocore/neurons/reference_trace_data/exact_current_lif_multitick_v1.json"
)


class TestExactLIFSolver:
    def test_spike_time_matches_analytical(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        # V_inf = -65 + 20 = -45 (above threshold)
        t = solver.next_spike_time(v0=-65.0, current=20.0)
        assert t is not None
        v_at_t = solver.evolve_to_time(-65.0, t, 20.0)
        assert abs(v_at_t - solver.v_thresh) < 1e-8

    def test_subthreshold_no_spike(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        t = solver.next_spike_time(v0=-65.0, current=10.0)
        assert t is None  # V_inf = -55, never reaches -50

    def test_already_threshold_spikes_immediately(self):
        solver = ExactLIFSolver(v_thresh=-50.0)

        assert solver.next_spike_time(v0=-50.0, current=20.0) == 0.0

    def test_evolve_to_time_at_zero(self):
        solver = ExactLIFSolver()
        v = solver.evolve_to_time(v0=-60.0, t=0.0, current=0.0)
        assert v == pytest.approx(-60.0)

    def test_subthreshold_evolution_is_bounded_by_equilibrium(self):
        solver = ExactLIFSolver(tau=20.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)

        voltage = solver.evolve_to_time(v0=-65.0, t=20.0, current=10.0)

        assert -65.0 < voltage < -55.0
        assert voltage < solver.v_thresh

    def test_firing_rate_suprathreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        rate = solver.firing_rate(current=30.0)
        assert rate > 0

    def test_firing_rate_subthreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        rate = solver.firing_rate(current=5.0)
        assert rate == 0.0

    def test_simulate_produces_spikes(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        spikes, _ = solver.simulate(current=30.0, t_end=100.0)
        assert len(spikes) >= 2

    def test_simulate_breaks_when_next_spike_exceeds_window(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)

        spikes, voltages = solver.simulate(current=20.0, t_end=1.0)

        assert spikes == []
        assert voltages == []

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau": 0.0}, "tau"),
            ({"tau": True}, "tau"),
            ({"tau": "bad"}, "tau"),
            ({"v_rest": float("nan")}, "v_rest"),
            ({"v_thresh": float("inf")}, "v_thresh"),
            ({"v_reset": -50.0, "v_thresh": -50.0}, "v_reset"),
            ({"r_m": 0.0}, "r_m"),
        ],
    )
    def test_rejects_invalid_physical_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ExactLIFSolver(**kwargs)

    @pytest.mark.parametrize(
        ("method", "args", "match"),
        [
            ("evolve_to_time", {"v0": float("nan"), "t": 1.0, "current": 1.0}, "v0"),
            ("evolve_to_time", {"v0": -65.0, "t": -1.0, "current": 1.0}, "t"),
            ("evolve_to_time", {"v0": -65.0, "t": True, "current": 1.0}, "t"),
            ("next_spike_time", {"v0": -65.0, "current": float("inf")}, "current"),
            ("simulate", {"current": 20.0, "t_end": -1.0}, "t_end"),
            ("simulate", {"current": True, "t_end": 1.0}, "current"),
            ("simulate", {"current": 20.0, "t_end": 1.0, "v0": float("nan")}, "v0"),
        ],
    )
    def test_rejects_invalid_runtime_inputs(self, method, args, match):
        solver = ExactLIFSolver()

        with pytest.raises(ValueError, match=match):
            getattr(solver, method)(**args)


class TestExactCurrentLIFProfile:
    _COMMIT = "1" * 40

    def test_profile_is_immutable_source_bound_and_canonical(self):
        profile = ExactCurrentLIFProfile()

        profile.verify_source_binding()
        restored = ExactCurrentLIFProfile.from_json(profile.to_json())

        assert restored == profile
        assert restored.digest == profile.digest
        assert len(profile.digest) == 64
        assert profile.to_json() == restored.to_json()
        with pytest.raises(FrozenInstanceError):
            profile.tau_ms = 1.0

    def test_profile_rejects_bound_source_drift(self, monkeypatch):
        profile = ExactCurrentLIFProfile()
        monkeypatch.setattr(lif_profile_module, "MODEL_SOURCE_SHA256", "0" * 64)

        with pytest.raises(ValueError, match="model source digest mismatch"):
            profile.verify_source_binding()

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            (lambda payload: payload.update(schema="future"), "unsupported profile"),
            (lambda payload: payload.update(extra=True), "fields mismatch"),
            (lambda payload: payload["units"].update(time="s"), "altered profile field: units"),
            (
                lambda payload: payload["model"].update(source_sha256="0" * 64),
                "altered profile field: model",
            ),
        ],
    )
    def test_profile_rejects_schema_fields_units_and_source_drift(self, mutation, match):
        payload = ExactCurrentLIFProfile().to_payload()
        mutation(payload)

        with pytest.raises(ValueError, match=match):
            ExactCurrentLIFProfile.from_json(json.dumps(payload))

    @pytest.mark.parametrize(
        "serialized",
        ["not-json", b"\xff", json.dumps([])],
    )
    def test_profile_rejects_malformed_json(self, serialized):
        with pytest.raises(ValueError):
            ExactCurrentLIFProfile.from_json(serialized)

    def test_profile_rejects_non_convertible_values(self):
        with pytest.raises(ValueError, match="tau_ms"):
            ExactCurrentLIFProfile(tau_ms=object())
        with pytest.raises(ValueError, match="tau_ms"):
            ExactCurrentLIFProfile(tau_ms=True)

    def test_profile_rejects_duplicate_json_members(self):
        with pytest.raises(ValueError, match="duplicate JSON field"):
            ExactCurrentLIFProfile.from_json('{"schema":"a","schema":"b"}')

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau_ms": 0.0}, "tau_ms"),
            ({"v_rest": -50.0}, "v_threshold"),
            ({"v_reset": -49.0}, "v_reset"),
            ({"resistance": float("inf")}, "resistance"),
        ],
    )
    def test_profile_rejects_invalid_parameter_domains(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ExactCurrentLIFProfile(**kwargs)

    def test_free_decay_matches_independent_closed_form(self):
        profile = ExactCurrentLIFProfile()
        session = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT)
        state = json.loads(session.serialize_state())
        state["state"]["voltage"] = -55.0
        session.restore_state(json.dumps(state))

        packet = session.execute([CurrentDriveTick(10.0, ())])
        expected = profile.v_rest + (-55.0 - profile.v_rest) * math.exp(-10.0 / profile.tau_ms)

        assert packet.events == ()
        assert packet.final_state.voltage == pytest.approx(expected, abs=1e-12)
        assert packet.state_trace[-1].phase == "tick_end"

    def test_exact_first_spike_threshold_reset_and_remaining_flow(self):
        profile = ExactCurrentLIFProfile(tau_ms=10.0)
        current = 30.0
        equilibrium = profile.v_rest + profile.resistance * current
        first_spike = -profile.tau_ms * math.log(
            (equilibrium - profile.v_threshold) / (equilibrium - profile.v_rest)
        )
        session = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT)

        packet = session.execute([CurrentDriveTick(first_spike + 1.0, (current,))])

        assert len(packet.events) == 1
        assert packet.events[0].time_ms == pytest.approx(first_spike, abs=1e-12)
        assert packet.events[0].voltage_before_reset == profile.v_threshold
        assert [sample.phase for sample in packet.state_trace] == [
            "initial",
            "threshold",
            "reset",
            "tick_end",
        ]
        expected_final = equilibrium + (profile.v_reset - equilibrium) * math.exp(
            -1.0 / profile.tau_ms
        )
        assert packet.final_state.voltage == pytest.approx(expected_final, abs=1e-12)

    def test_threshold_equality_emits_at_exact_tick_end(self):
        profile = ExactCurrentLIFProfile(tau_ms=10.0)
        current = 30.0
        equilibrium = profile.v_rest + profile.resistance * current
        crossing = -profile.tau_ms * math.log(
            (equilibrium - profile.v_threshold) / (equilibrium - profile.v_rest)
        )
        session = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT)

        packet = session.execute([CurrentDriveTick(crossing, (current,))])

        assert [event.time_ms for event in packet.events] == pytest.approx([crossing], abs=1e-12)
        assert packet.final_state.voltage == profile.v_reset

    def test_simultaneous_inputs_and_multi_call_state_are_order_independent(self):
        profile = ExactCurrentLIFProfile()
        split = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT)
        first = split.execute([CurrentDriveTick(1.0, (10.0, 20.0))])
        second = split.execute([CurrentDriveTick(1.0, (20.0, 10.0))])
        combined = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT).execute(
            [CurrentDriveTick(2.0, (30.0,))]
        )

        assert second.initial_state == first.final_state
        assert second.final_state.voltage == pytest.approx(combined.final_state.voltage, abs=1e-12)
        assert second.final_state.time_ms == 2.0
        assert first.ticks[0].total_current == second.ticks[0].total_current == 30.0

    def test_state_restore_replay_is_digest_bound_and_deterministic(self):
        profile = ExactCurrentLIFProfile()
        original = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT, shot_id="shot-a")
        original.execute([CurrentDriveTick(3.0, (25.0,))])
        checkpoint = original.serialize_state()
        expected = original.execute([CurrentDriveTick(4.0, (25.0,))])
        replay = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT, shot_id="unused")

        replay.restore_state(checkpoint)
        observed = replay.execute([CurrentDriveTick(4.0, (25.0,))])

        assert observed.to_json() == expected.to_json()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("schema", "future", "unsupported state schema"),
            ("profile_sha256", "0" * 64, "digest mismatch"),
        ],
    )
    def test_restore_rejects_version_and_digest_without_mutation(self, field, value, match):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        payload = json.loads(session.serialize_state())
        payload[field] = value

        with pytest.raises(ValueError, match=match):
            session.restore_state(json.dumps(payload))

        assert session.state == before

    def test_restore_rejects_unknown_fields_and_threshold_state_atomically(self):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        payload = json.loads(session.serialize_state())
        payload["state"]["unknown"] = 1
        with pytest.raises(ValueError, match="fields mismatch"):
            session.restore_state(json.dumps(payload))
        payload["state"].pop("unknown")
        payload["state"]["voltage"] = session.profile.v_threshold
        with pytest.raises(ValueError, match="below threshold"):
            session.restore_state(json.dumps(payload))
        assert session.state == before

    def test_shot_reset_is_explicit_and_increments_epoch(self):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        session.execute([CurrentDriveTick(2.0, (10.0,))])

        reset = session.reset_shot("shot-b")

        assert reset.voltage == session.profile.v_rest
        assert reset.time_ms == 0.0
        assert reset.shot_id == "shot-b"
        assert reset.reset_epoch == 1

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("time_ms", -1.0, "non-negative"),
            ("reset_epoch", True, "non-negative integer"),
            ("reset_epoch", -1, "non-negative integer"),
            ("reset_epoch", 1.5, "non-negative integer"),
            ("shot_id", "x" * 129, "at most 128"),
        ],
    )
    def test_restore_rejects_invalid_complete_state_fields(self, field, value, match):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        payload = json.loads(session.serialize_state())
        payload["state"][field] = value

        with pytest.raises(ValueError, match=match):
            session.restore_state(json.dumps(payload))

        assert session.state == before

    def test_packet_is_canonical_and_binds_execution_provenance(self):
        profile = ExactCurrentLIFProfile()
        packet = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT).execute([])
        payload = json.loads(packet.to_json())

        assert payload["producer_commit"] == self._COMMIT
        assert payload["profile"]["sha256"] == profile.digest
        assert payload["solver"] == "closed_form_piecewise_constant_event_driven"
        assert payload["rng"] == "none"
        assert payload["reset_boundary"] == "explicit_shot_reset_only"
        assert packet.initial_state == packet.final_state
        assert (
            type(packet).from_json(
                packet.to_json(), profile=profile, expected_producer_commit=self._COMMIT
            )
            == packet
        )

    def test_immutable_multitick_receipt_matches_independent_oracle(self):
        implementation_commit = "bc76e5b3c217fec191534bb650685316e645ad34"
        profile_json = _PROFILE_RECEIPT.read_text(encoding="utf-8").strip()
        profile = ExactCurrentLIFProfile.from_json(profile_json)

        assert profile.to_json() == profile_json
        assert profile.digest == "c667f3885f564dcf968febaf62125a86abaaee4758df792d5f06b0e82d1f121a"
        packet = ExactLIFExecutionPacket.from_json(
            _PACKET_RECEIPT.read_text(encoding="utf-8"),
            profile=profile,
            expected_producer_commit=implementation_commit,
        )
        assert len(packet.ticks) == 4
        assert [len(tick.currents) for tick in packet.ticks] == [1, 2, 2, 1]
        assert [event.tick for event in packet.events] == [1, 3, 3]
        assert [sample.phase for sample in packet.state_trace] == [
            "initial",
            "tick_end",
            "threshold",
            "reset",
            "tick_end",
            "tick_end",
            "threshold",
            "reset",
            "threshold",
            "reset",
            "tick_end",
        ]

        def evolve(voltage: float, duration_ms: float, current: float) -> float:
            equilibrium = profile.v_rest + profile.resistance * current
            return equilibrium + (voltage - equilibrium) * math.exp(-duration_ms / profile.tau_ms)

        def crossing(voltage: float, current: float) -> float:
            equilibrium = profile.v_rest + profile.resistance * current
            return -profile.tau_ms * math.log(
                (equilibrium - profile.v_threshold) / (equilibrium - voltage)
            )

        after_tick_0 = evolve(profile.v_rest, 5.0, 10.0)
        crossing_1 = 5.0 + crossing(after_tick_0, 30.0)
        after_tick_1 = evolve(profile.v_reset, 25.0 - crossing_1, 30.0)
        after_tick_2 = evolve(after_tick_1, 7.0, 0.0)
        crossing_2 = 32.0 + crossing(after_tick_2, 30.0)
        reset_isi = crossing(profile.v_reset, 30.0)
        crossing_3 = crossing_2 + reset_isi
        final_voltage = evolve(profile.v_reset, 62.0 - crossing_3, 30.0)

        assert [event.time_ms for event in packet.events] == pytest.approx(
            [crossing_1, crossing_2, crossing_3], abs=1e-12
        )
        assert packet.final_state.voltage == pytest.approx(final_voltage, abs=1e-12)

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda payload: payload.update(schema="future"), "unsupported packet schema"),
            (lambda payload: payload.update(extra=True), "fields mismatch"),
            (lambda payload: payload.update(producer_commit="2" * 40), "producer commit mismatch"),
            (
                lambda payload: payload["profile"].update(sha256="0" * 64),
                "profile binding mismatch",
            ),
            (
                lambda payload: payload["state_trace"][0].update(voltage=-64.0),
                "failed deterministic replay",
            ),
            (
                lambda payload: payload["ticks"][0].update(total_current=999.0),
                "total_current mismatch",
            ),
        ],
    )
    def test_packet_parser_rejects_schema_provenance_and_trace_tampering(self, mutate, match):
        profile = ExactCurrentLIFProfile()
        packet = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT).execute(
            [CurrentDriveTick(1.0, (2.0, 3.0))]
        )
        payload = packet.to_payload()
        mutate(payload)

        with pytest.raises(ValueError, match=match):
            type(packet).from_json(
                json.dumps(payload), profile=profile, expected_producer_commit=self._COMMIT
            )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("ticks", {}, "ticks must be an array"),
            ("currents", {}, "tick currents must be an array"),
        ],
    )
    def test_packet_parser_rejects_non_array_inputs(self, field, value, match):
        profile = ExactCurrentLIFProfile()
        packet = ExactCurrentLIFSession(profile, producer_commit=self._COMMIT).execute(
            [CurrentDriveTick(1.0, (2.0,))]
        )
        payload = packet.to_payload()
        if field == "ticks":
            payload[field] = value
        else:
            payload["ticks"][0][field] = value

        with pytest.raises(ValueError, match=match):
            type(packet).from_json(
                json.dumps(payload), profile=profile, expected_producer_commit=self._COMMIT
            )

    def test_packet_parser_rejects_duplicate_and_malformed_json(self):
        profile = ExactCurrentLIFProfile()
        with pytest.raises(ValueError, match="duplicate JSON field"):
            ExactLIFExecutionPacket.from_json(
                '{"schema":"a","schema":"b"}',
                profile=profile,
                expected_producer_commit=self._COMMIT,
            )
        with pytest.raises(ValueError, match="valid JSON"):
            ExactLIFExecutionPacket.from_json(
                "not-json",
                profile=profile,
                expected_producer_commit=self._COMMIT,
            )

    def test_execute_is_failure_atomic_on_late_time_overflow(self):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state

        with pytest.raises(ValueError, match="execution time overflowed"):
            session.execute(
                [
                    CurrentDriveTick(1e308, (0.0,)),
                    CurrentDriveTick(1e308, (0.0,)),
                ]
            )

        assert session.state == before

    def test_tick_rejects_non_finite_simultaneous_sum(self):
        with pytest.raises(ValueError, match="summed current"):
            CurrentDriveTick(1.0, (1e308, 1e308))

    def test_tick_rejects_non_finite_sum_result(self, monkeypatch):
        monkeypatch.setattr(lif_profile_module.math, "fsum", lambda _: float("inf"))

        with pytest.raises(ValueError, match="summed current"):
            CurrentDriveTick(1.0, (1.0,))

    def test_execute_rejects_non_positive_crossing_atomically(self, monkeypatch):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        monkeypatch.setattr(lif_profile_module.ExactLIFSolver, "next_spike_time", lambda *_: 0.0)

        with pytest.raises(FloatingPointError, match="non-positive"):
            session.execute([CurrentDriveTick(1.0, (30.0,))])

        assert session.state == before

    def test_execute_rejects_non_finite_crossing_atomically(self, monkeypatch):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        monkeypatch.setattr(
            lif_profile_module.ExactLIFSolver, "next_spike_time", lambda *_: float("nan")
        )

        with pytest.raises(FloatingPointError, match="non-finite"):
            session.execute([CurrentDriveTick(1.0, (30.0,))])

        assert session.state == before

    def test_execute_rejects_non_finite_evolution_atomically(self, monkeypatch):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        monkeypatch.setattr(lif_profile_module.ExactLIFSolver, "next_spike_time", lambda *_: None)
        monkeypatch.setattr(
            lif_profile_module.ExactLIFSolver, "evolve_to_time", lambda *_: float("inf")
        )

        with pytest.raises(FloatingPointError, match="non-finite voltage"):
            session.execute([CurrentDriveTick(1.0, (1.0,))])

        assert session.state == before

    def test_execute_rejects_crossing_without_clock_progress_atomically(self, monkeypatch):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        state = json.loads(session.serialize_state())
        state["state"]["time_ms"] = 1e20
        session.restore_state(json.dumps(state))
        before = session.state
        monkeypatch.setattr(lif_profile_module.ExactLIFSolver, "next_spike_time", lambda *_: 1.0)

        with pytest.raises(FloatingPointError, match="no finite progress"):
            session.execute([CurrentDriveTick(1e6, (30.0,))])

        assert session.state == before

    def test_execute_enforces_event_bound_atomically(self, monkeypatch):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state
        monkeypatch.setattr(lif_profile_module, "_MAX_EVENTS_PER_TICK", 0)

        with pytest.raises(ValueError, match="event count exceeded"):
            session.execute([CurrentDriveTick(100.0, (30.0,))])

        assert session.state == before

    @pytest.mark.parametrize(
        ("factory", "match"),
        [
            (lambda: CurrentDriveTick(0.0, (1.0,)), "duration_ms"),
            (lambda: CurrentDriveTick(1.0, (float("nan"),)), "current"),
            (
                lambda: ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit="bad"),
                "producer_commit",
            ),
            (
                lambda: ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=True),
                "producer_commit",
            ),
            (
                lambda: ExactCurrentLIFSession(
                    ExactCurrentLIFProfile(), producer_commit="1" * 40, shot_id=""
                ),
                "shot_id",
            ),
        ],
    )
    def test_public_contract_rejects_invalid_inputs(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()

    def test_execute_rejects_non_tick_values_without_mutation(self):
        session = ExactCurrentLIFSession(ExactCurrentLIFProfile(), producer_commit=self._COMMIT)
        before = session.state

        with pytest.raises(TypeError, match="CurrentDriveTick"):
            session.execute([object()])

        assert session.state == before

    def test_session_rejects_wrong_profile_type(self):
        with pytest.raises(TypeError, match="ExactCurrentLIFProfile"):
            ExactCurrentLIFSession(object(), producer_commit=self._COMMIT)

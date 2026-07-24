# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (monitor_and_dispatch) from former test_network_execution_paths.py

from __future__ import annotations

from tests.network_execution_paths_support import *  # noqa: F403


def test_auto_backend_uses_python_for_state_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = StateMonitor(pop, ["v"])

    Network(pop, monitor).run(0.003, dt=0.001, backend="auto")

    assert fake_runner.instances == []
    assert monitor.traces["v"].shape == (3, 2)
    assert monitor.t.tolist() == [0, 1, 2]


def test_forced_rust_rejects_state_monitors_until_step_traces_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = StateMonitor(pop, ["v"])

    with pytest.raises(NotImplementedError, match="StateMonitor"):
        Network(pop, monitor).run(0.003, dt=0.001, backend="rust")


def test_forced_rust_rejects_rate_monitors_until_step_traces_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)
    monitor = RateMonitor(pop)

    with pytest.raises(NotImplementedError, match="RateMonitor"):
        Network(pop, monitor).run(0.003, dt=0.001, backend="rust")


def test_auto_backend_uses_python_for_spike_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    Network(pop).run(0.003, dt=0.001, backend="auto", spike_gating=True)

    assert fake_runner.instances == []


def test_forced_rust_rejects_spike_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    with pytest.raises(NotImplementedError, match="spike_gating"):
        Network(pop).run(0.003, dt=0.001, backend="rust", spike_gating=True)


def test_auto_backend_uses_python_for_fim_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    Network(pop, fim_lambda=1.0).run(0.003, dt=0.001, backend="auto")

    assert fake_runner.instances == []


def test_forced_rust_rejects_fim_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2)

    with pytest.raises(NotImplementedError, match="fim_lambda"):
        Network(pop, fim_lambda=1.0).run(0.003, dt=0.001, backend="rust")


def test_auto_rust_dispatch_uses_model_identity_not_population_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = _install_fake_rust_engine(monkeypatch)
    pop = Population(_MODEL, 2, label="exc")

    Network(pop).run(0.003, dt=0.001, backend="auto")

    assert len(fake_runner.instances) == 1
    assert fake_runner.instances[0].added_models == [_MODEL]

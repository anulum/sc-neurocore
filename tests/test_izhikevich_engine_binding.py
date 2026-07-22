# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich engine-binding contracts

"""Installed-extension contracts for the floating-point Izhikevich neuron."""

from __future__ import annotations

import importlib

import pytest

import sc_neurocore_engine as engine

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_class_signature_and_top_level_identity_are_stable() -> None:
    izhikevich = extension.Izhikevich

    assert izhikevich.__name__ == "Izhikevich"
    assert izhikevich.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    # PyO3 renders the signed numeric default as an ellipsis in the class
    # text signature; the constructor-state assertion below pins its value.
    assert izhikevich.__text_signature__ == "(a=0.02, b=0.2, c=..., d=8.0, dt=1.0)"
    assert engine.Izhikevich is izhikevich

    default_state = izhikevich().get_state()
    assert default_state == pytest.approx({"v": -65.0, "u": -13.0})


def _reference_step(
    v: float,
    u: float,
    *,
    current: float,
    a: float,
    b: float,
    c: float,
    d: float,
    dt: float,
) -> tuple[int, float, float]:
    half = dt * 0.5
    for _ in range(2):
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + current) * half
        du = a * (b * v - u) * half
        v += dv
        u += du
    if v >= 30.0:
        return 1, c, u + d
    return 0, v, u


def test_step_trace_matches_independent_two_half_step_recurrence() -> None:
    parameters = {"a": 0.03, "b": 0.22, "c": -61.0, "d": 5.0, "dt": 0.5}
    neuron = extension.Izhikevich(**parameters)
    v = parameters["c"]
    u = parameters["b"] * parameters["c"]

    for current in (0.0, 8.0, 12.0, 20.0) * 12:
        expected_spike, v, u = _reference_step(v, u, current=current, **parameters)
        assert neuron.step(current) == expected_spike
        state = neuron.get_state()
        assert state["v"] == pytest.approx(v, rel=0.0, abs=1e-12)
        assert state["u"] == pytest.approx(u, rel=0.0, abs=1e-12)


def test_reset_and_reset_state_restore_constructor_state() -> None:
    neuron = extension.Izhikevich(a=0.1, b=0.25, c=-58.0, d=2.0, dt=0.25)

    for _ in range(16):
        neuron.step(25.0)
    neuron.reset()
    assert neuron.get_state() == pytest.approx({"v": -58.0, "u": -14.5})

    neuron.step(10.0)
    neuron.reset_state()
    assert neuron.get_state() == pytest.approx({"v": -58.0, "u": -14.5})

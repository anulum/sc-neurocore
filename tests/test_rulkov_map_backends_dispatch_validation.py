# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Rulkov backend contracts

"""Focused cross-backend Rulkov map contracts."""

from .rulkov_map_backends_support import *

from sc_neurocore.neurons.models import rulkov_map as implementation


def test_auto_matches_python() -> None:
    ref, ref_spikes, _rx, _ry = _run("python")
    got, spikes, _xf, _yf = _run("auto")
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0.0)
    assert spikes == ref_spikes


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        RulkovMapNeuron().simulate(10, 0.0, backend="cuda")


def test_negative_n_steps_raises() -> None:
    with pytest.raises(ValueError, match="n_steps must be between"):
        RulkovMapNeuron().simulate(-1, 0.0)


@pytest.mark.parametrize("n_steps", (True, 2**31))
def test_invalid_step_count_contract_raises(n_steps: int) -> None:
    with pytest.raises(ValueError, match="n_steps must"):
        RulkovMapNeuron().simulate(n_steps, 0.0)


def test_non_finite_current_raises() -> None:
    with pytest.raises(ValueError, match="current must be finite"):
        RulkovMapNeuron().simulate(10, np.nan)


@pytest.mark.parametrize(
    ("backend", "runtime_attribute"),
    [
        ("rust", "_rust_simulate"),
        ("julia", "_julia_module"),
        ("go", "_go_lib"),
        ("mojo", "_mojo_lib"),
    ],
)
def test_runtime_disappearance_fails_closed_without_state_mutation(
    monkeypatch: pytest.MonkeyPatch, backend: str, runtime_attribute: str
) -> None:
    """A runtime lost after selection must fail before public-state commit."""

    def report_available(_backend: str) -> bool:
        return True

    monkeypatch.setattr(implementation, "_backend_available", report_available)
    monkeypatch.setattr(implementation, runtime_attribute, None)
    neuron = RulkovMapNeuron()
    before = (neuron.x, neuron.y)

    with pytest.raises(RuntimeError, match=f"{backend} Rulkov backend is unavailable"):
        neuron.simulate(1, 0.5, backend=backend)

    assert (neuron.x, neuron.y) == before

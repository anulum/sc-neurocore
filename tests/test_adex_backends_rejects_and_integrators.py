# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_integrators) from former test_adex_backends.py

from __future__ import annotations

from tests.adex_backends_support import *  # noqa: F403

@pytest.mark.parametrize("backend", _COMPILED_BACKENDS)
def test_compiled_backends_reject_non_baseline_integrators(backend: str) -> None:
    """Never silently run baseline Euler for an RK4-configured instance."""
    neuron = AdExNeuron(integrator="rk4")
    before = (neuron.v, neuron.w)
    with pytest.raises(RuntimeError, match="baseline_euler"):
        neuron.simulate(1, 0.0, backend=backend)
    assert (neuron.v, neuron.w) == before


def test_rust_rejects_non_default_contract() -> None:
    """Keep the engine class's factory-only parameter boundary explicit."""
    neuron = AdExNeuron(v=-60.0)
    before = (neuron.v, neuron.w)
    with pytest.raises(RuntimeError, match="factory-default"):
        neuron.simulate(1, 0.0, backend="rust")
    assert (neuron.v, neuron.w) == before


def test_auto_uses_python_for_alternative_integrators() -> None:
    """Keep optional RK4 and Rosenbrock semantics independent of baseline kernels."""
    auto = AdExNeuron(integrator="rk4")
    python = AdExNeuron(integrator="rk4")
    auto_trace, auto_spikes = auto.simulate(100, 250.0, backend="auto")
    python_trace, python_spikes = python.simulate(100, 250.0, backend="python")
    np.testing.assert_array_equal(auto_trace, python_trace)
    assert (auto_spikes, auto.v, auto.w) == (python_spikes, python.v, python.w)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Hindmarsh-Rose backend test support

"""Backend discovery and simulation fixtures for Hindmarsh–Rose parity tests."""

from __future__ import annotations

from sc_neurocore.neurons.models import hindmarsh_rose as hr
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron

_STEP_TOL = 8e-15  # per-step Mojo FMA bound (measured worst ~1.8e-15 over x/y/z)


def _run(backend: str, *, current: float = 3.0, n: int = 8000, **kw) -> tuple:  # type: ignore[type-arg, no-untyped-def] # Preserved legacy helper AST
    neuron = HindmarshRoseNeuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.x, neuron.y, neuron.z


def _rust() -> bool:
    return hr._HAS_RUST


def _julia() -> bool:
    return hr._ensure_julia_loaded()


def _go() -> bool:
    return hr._ensure_go_loaded()


def _mojo() -> bool:
    return hr._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 1.0, 2.0, 3.2]

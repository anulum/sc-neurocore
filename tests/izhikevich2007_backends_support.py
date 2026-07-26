# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Izhikevich-2007 backend test support

"""Backend discovery and simulation fixtures for Izhikevich-2007 parity."""

from __future__ import annotations

from sc_neurocore.neurons.models import izhikevich2007 as izh
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron

# Mojo FMA band: ~5e-12 at strong drive, up to ~4e-8 when firing is sparse.
_MOJO_ATOL = 1e-6


def _run(backend: str, *, current: float = 300.0, n: int = 8000, **kw) -> tuple:  # type: ignore[type-arg, no-untyped-def] # Preserved legacy helper AST
    neuron = Izhikevich2007Neuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.u


def _rust() -> bool:
    return izh._HAS_RUST


def _julia() -> bool:
    return izh._ensure_julia_loaded()


def _go() -> bool:
    return izh._ensure_go_loaded()


def _mojo() -> bool:
    return izh._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 100.0, 300.0, 500.0]

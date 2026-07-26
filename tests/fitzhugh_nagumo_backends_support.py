# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared FitzHugh-Nagumo backend test support

"""Backend discovery and simulation fixtures for FitzHugh–Nagumo parity."""

from __future__ import annotations

from sc_neurocore.neurons.models import fitzhugh_nagumo as fhn
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron

# Mojo FMA band (measured ~6e-15 over 8k steps); generous and non-amplifying.
_MOJO_ATOL = 1e-11


def _run(backend: str, *, current: float = 0.5, n: int = 8000, **kw) -> tuple:  # type: ignore[type-arg, no-untyped-def] # Preserved legacy helper AST
    neuron = FitzHughNagumoNeuron(**kw)
    trace, spikes = neuron.simulate(n, current, backend=backend)
    return trace, spikes, neuron.v, neuron.w


def _rust() -> bool:
    return fhn._HAS_RUST


def _julia() -> bool:
    return fhn._ensure_julia_loaded()


def _go() -> bool:
    return fhn._ensure_go_loaded()


def _mojo() -> bool:
    return fhn._ensure_mojo_loaded()


_BIT_EXACT = [("rust", _rust), ("julia", _julia), ("go", _go)]
_CURRENTS = [0.0, 0.3, 0.5, 1.0]

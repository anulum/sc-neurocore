# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge wrapper for NetworkRunner

"""Stable bridge wrapper for the Rust `NetworkRunner` surface."""

from __future__ import annotations

from typing import Any


def get_network_runner_class() -> Any:
    """Return the Rust `NetworkRunner` class or raise ImportError."""
    from sc_neurocore_engine.sc_neurocore_engine import NetworkRunner

    return NetworkRunner

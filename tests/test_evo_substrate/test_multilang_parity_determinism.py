# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolution runner fixed-seed determinism test

"""Verify repeatable Rust evolution results under a fixed seed."""

from __future__ import annotations

import json
from typing import Any, Protocol, cast


pytest_plugins = ["tests.test_evo_substrate.multilang_parity_support"]

JsonObject = dict[str, Any]


class EvoRunner(Protocol):
    """Describe the Rust runner contract supplied by the shared fixture."""

    def py_evolve_run(self, config_json: str) -> str:
        """Run one evolution configuration and return JSON."""
        ...


def test_rust_seed_determinism(cfg_json: str, rust_runner_backend: EvoRunner) -> None:
    first = cast(JsonObject, json.loads(rust_runner_backend.py_evolve_run(cfg_json)))
    second = cast(JsonObject, json.loads(rust_runner_backend.py_evolve_run(cfg_json)))
    assert first == second, "Rust runner is non-deterministic under fixed seed"
